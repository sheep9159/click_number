import os
import struct
import numpy as np
import pandas as pd
import scipy.signal as signal
from sklearn.decomposition import FastICA
from sklearn import preprocessing


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


def get_eeg_is_useful_from_psd(eeg_data, Fs=500, win='hamming', scal='spectrum', high_frequency=42, threshold=0.5):
    """
    此函数通过得到信号频谱图，进而计算高频所占能量比例，如果高于threshold，则返回False
    :param eeg_data:输入的多通道eeg信号
    :param Fs:eeg信号的采样频率
    :param win:使用的窗口函数
    :param scal:使用的是功率谱还是功率谱密度
    :param high_frequency:高于high_frequency的频段算作高频，默认是42hz
    :param threshold:设定的能量占比阈值
    :return:返回布尔值，表明所检验的eeg信号是否可用
    """
    channels, points = eeg_data.shape
    each_channel_high_frequency_percent = np.zeros(channels)

    for channel in range(channels):
        f, Pper_spec = signal.periodogram(eeg_data[channel], fs=Fs, window=win, scaling=scal)
        spec = Pper_spec.sum()  # 得到信号能量总和
        high_spec = Pper_spec[len(f[f<42]):].sum()  # 得到频率大于high_frequency的频谱能量总和
        each_channel_high_frequency_percent[channel] = high_spec / spec  # 得到某一通道下的高频能量占比

    high_frequency_percent = each_channel_high_frequency_percent.sum() / channels  # 得到所有通道高频能量占比的平均值
    if high_frequency_percent > threshold:
        return False
    else:
        return True


if __name__ == '__main__':

    import data_dir

    num = 0  # 用于记录所有被使者的所有有效的脑电信号被分割成多少段

    index = ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F8', 'F3', 'Fz', 'F4', 'FC5', 'FC6', 'T7', 'T8', 'C3',
               'Cz', 'C4', 'CP5', 'CP6', 'P7', 'P8', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'PO3', 'PO4', 'O1',
               'O2']

    mark_file_name = get_file_name(data_dir.raw_eeg_dir, '.vmrk')
    eeg_file_name = get_file_name(data_dir.raw_eeg_dir, '.eeg')

    for i in range(len(mark_file_name)):  # 逐一对文件夹中的数据进行处理

        mark, point = read_mark_txt(mark_file_name[i])
        raw_eeg = read_eeg_file(eeg_file_name[i])

        # 第一次运行的时候用于保存最原始的分割后的数据，没有做任何操作，如去除坏道，去除高频干扰过大的数据
        # pd.DataFrame(raw_eeg, index=index).to_csv(raw_eeg_dir + r'\raw_eeg\{}.csv'.format(i))

        for j in range(len(mark) - 1):
            if mark[j+1] != 0:  # 将每个子试验分割开来，子试验间的空隙测量到的脑电信号属无用数据，应删除
                data = raw_eeg[:, point[j]: point[j + 1]]  # 将脑电信号转换成(35, points)的矩阵
                data = data[:29, :]  # 舍去最后一个电极数据，因为实验室的脑电帽这个电极是坏的
                if get_eeg_is_useful_from_psd(data):
                    zero = np.zeros_like(data)
                    if not (data < zero).any():  # 只要数据当中又负值，则认为存在坏道，应去除
                        num += 1

                        ica = FastICA(n_components=29, max_iter=1000, tol=0.05)
                        data = ica.fit_transform(data.transpose())
                        data = data.transpose()
                        data = preprocessing.scale(data, axis=1)

                        if (point[j+1] - point[j]) < 250:  # 500hz采样率，则250代表反应时间在0.5s
                            df_label1 = pd.DataFrame(data, index=index)
                            df_label1.to_csv(data_dir.preprocess_dir + r'\level1\{}.csv'.format(num))
                        elif (point[j+1] - point[j]) > 250 and (point[j+1] - point[j]) < 750:  # 0.5s-1.5s
                            df_label2 = pd.DataFrame(data, index=index)
                            df_label2.to_csv(data_dir.preprocess_dir + r'\level2\{}.csv'.format(num))
                        else:                                                                  # >1.5s
                            df_label3 = pd.DataFrame(data, index=index)
                            df_label3.to_csv(data_dir.preprocess_dir + r'\level3\{}.csv'.format(num))