import filter
import os
import struct
import numpy as np
import pandas as pd
import scipy.signal as signal


def get_file_name(file_dir, file_type):
    """
    :遍历指定目录下的所有指定类型的数据文件
    :file_dir: 此目录下包含.eeg原始数据文件，.vhdr文件(含mark)和.vmrk文件
    :file_type: 指定需要找到的文件类型

    :返回
    :file_names: 指定文件的绝对路径
    """

    file_names = []

    for root, dirs, files in os.walk(file_dir, topdown=False):
        for file in files:
            if file_type in file:
                file_names.append(os.path.join(root, file))

    return file_names


def read_mark_txt(file_dir):
    """
    :读取.vhdr文件中mark信息
    :file_dir: 是.vhdr文件的绝对路径

    :返回
    :mark: 点击数字时发送到脑电帽的一个mark标记
    :point: 代表点击数字时已经记录了多少次脑电信号
    """
    mark = []
    point = []
    with open(file_dir, 'r') as f:
        for line in f:
            if line[0:2] == 'Mk':
                mark.append(int(line[3:].split(',')[1]))
                point.append(int(line[3:].split(',')[2]))

    return mark[1:], point[1:]  # 从试验一开始记录脑电信号到第一个点数字子试验开始，这之间是无用的脑电信号，应删除


def read_eeg_file(file_dir):
    """
    :读取eeg原始数据
    """
    with open(file_dir, 'rb') as raw:
        raw = raw.read()
        raw_eeg = struct.unpack('{}f'.format(int(len(raw) / 4)), raw)
        raw_eeg = np.array(raw_eeg).reshape((35, -1), order='F')  # 指定每35个数据得到一列

    return raw_eeg


num = 0  # 用于记录所有被使者的所有脑电信号被分割成多少段

index = ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F8', 'F3', 'Fz', 'F4', 'FC5', 'FC6', 'T7', 'T8', 'C3',
           'Cz', 'C4', 'CP5', 'CP6', 'P7', 'P8', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'PO3', 'PO4', 'O1',
           'O2', 'A2', 'ACC30', 'ACC31', 'ACC31', 'Packet Counter', 'TRIGGER']

mark_file_name = get_file_name(r'D:\Files\SJTU\Study\MME_Lab\Teacher_Lu\click_number\eeg\raw_data', '.vmrk')
eeg_file_name = get_file_name(r'D:\Files\SJTU\Study\MME_Lab\Teacher_Lu\click_number\eeg\raw_data', '.eeg')

for i in range(len(mark_file_name)):  # 逐一对文件夹中的数据进行处理

    mark, point = read_mark_txt(mark_file_name[i])
    raw_eeg = read_eeg_file(eeg_file_name[i])

    # pd.DataFrame(raw_eeg, index=index).to_csv(r'D:\Files\SJTU\Study\MME_Lab\Teacher_Lu\click_number\eeg\raw_eeg\{}.csv'.format(i))

    for j in range(len(mark) - 1):
        if mark[j+1] != 0:  # 将每个子试验分割开来，子试验间的空隙测量到的脑电信号属无用数据，应删除
            data = raw_eeg[:, point[j]: point[j + 1]]  # 将脑电信号转换成(35, points)的矩阵
            if filter.get_eeg_is_useful_from_psd(data):
                num += 1
                if (point[j+1] - point[j]) < 250:  # 500hz采样率，则250秒代表反应时间在0.5秒
                    df_label1 = pd.DataFrame(data, index=index)
                    df_label1.to_csv(r'D:\Files\SJTU\Study\MME_Lab\Teacher_Lu\click_number\eeg\process3.0\label1\{}.csv'.format(num))
                elif (point[j+1] - point[j]) > 250 and (point[j+1] - point[j]) < 750:
                    df_label2 = pd.DataFrame(data, index=index)
                    df_label2.to_csv(r'D:\Files\SJTU\Study\MME_Lab\Teacher_Lu\click_number\eeg\process3.0\label2\{}.csv'.format(num))
                else:
                    df_label3 = pd.DataFrame(data, index=index)
                    df_label3.to_csv(r'D:\Files\SJTU\Study\MME_Lab\Teacher_Lu\click_number\eeg\\process3.0\label3\{}.csv'.format(num))

            #**************************************重新划分脑电信号*******************************************************#

            # if (point[j+1] - point[j]) <= 200:  # 0.4s
            #     fast_reaction = raw_eeg[:, point[j] : point[j+1]]
            #     df_fast = pd.DataFrame(fast_reaction, index=index)
            #     df_fast.to_csv(r'D:\Files\SJTU\Study\MME_Lab\Teacher_Lu\click_number\eeg\process2.0\label1\{}.csv'.format(num))
            # elif (point[j+1] - point[j]) > 200 and (point[j+1] - point[j]) <= 350:  # 0.4s-0.7s
            #     medium_reaction = raw_eeg[:, point[j] : point[j + 1]]
            #     df_medium = pd.DataFrame(medium_reaction, index=index)
            #     df_medium.to_csv(r'D:\Files\SJTU\Study\MME_Lab\Teacher_Lu\click_number\eeg\process2.0\label2\{}.csv'.format(num))
            # elif (point[j+1] - point[j]) > 350 and (point[j+1] - point[j]) <= 500:  # 0.7s-1s
            #     slow_reaction = raw_eeg[:, point[j] : point[j + 1]]
            #     df_slow = pd.DataFrame(slow_reaction, index=index)
            #     df_slow.to_csv(r'D:\Files\SJTU\Study\MME_Lab\Teacher_Lu\click_number\eeg\process2.0\label3\{}.csv'.format(num))
            # elif (point[j+1] - point[j]) > 500 and (point[j+1] - point[j]) <= 650:  # 1s-1.3s
            #     slow_reaction = raw_eeg[:, point[j] : point[j + 1]]
            #     df_slow = pd.DataFrame(slow_reaction, index=index)
            #     df_slow.to_csv(r'D:\Files\SJTU\Study\MME_Lab\Teacher_Lu\click_number\eeg\process2.0\label4\{}.csv'.format(num))
            # else:
            #     slow_reaction = raw_eeg[:, point[j] : point[j + 1]]
            #     df_slow = pd.DataFrame(slow_reaction, index=index)
            #     df_slow.to_csv(r'D:\Files\SJTU\Study\MME_Lab\Teacher_Lu\click_number\eeg\process2.0\label5\{}.csv'.format(num))