import data_dir
import segmentation
import featureExtract

import os
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
import re
import mne


WINDOWS = 250  # 指定窗口长度，用于分割序列，观察在相同的时间跨度上的相似性

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
    :读取.vmrk文件中mark信息
    :file_dir: 是.vmrk文件的绝对路径

    :返回
    :mark: 点击数字时发送到脑电帽的一个mark标记
    :point: 代表点击数字时已经记录了多少次脑电信号
    """
    mark = []
    point = []
    with open(file_dir, 'r') as f:
        for line in f:
            if line[0:2] == 'MK':
                mark.append(int(line[3:].split(',')[1]))
                point.append(int(line[3:].split(',')[2]))

    return mark, point


if __name__ == '__main__':

    # eng = matlab.engine.start_matlab()
    #
    # # 从原数据中分段
    # mark_file_name = get_file_name(data_dir.raw_eeg_dir, '.vmrk')
    # vhdr_file_name = get_file_name(data_dir.raw_eeg_dir, '.vhdr')
    # eeg_file_name = get_file_name(data_dir.raw_eeg_dir, '.eeg')
    #
    # for i in range(len(eeg_file_name)):
    #
    #     point_boundary, name_index, trail_indices, datas_length = segmentation.read_mark_txt(mark_file_name[i])
    #     segmentation.read_vhdr_file(vhdr_file_name[i], name_index, trail_indices, datas_length)
    #     segmentation.read_eeg_file(eeg_file_name[i], point_boundary, name_index, trail_indices)
    #
    #
    # # eeglab进行第一步处理
    # eng.batch_preprocess1(nargout=0)
    #
    #
    # # 找到bad_channel
    # set_file_name = get_file_name(data_dir.segmentation_dir+r'\preprocess1', '.set')
    #
    # for index in range(len(set_file_name)):
    #     pattern = re.compile('\w+\.set')
    #     curren_name_index = pattern.findall(set_file_name[index])[0][:-4]
    #
    #     raw_eeg = mne.io.read_raw_eeglab(set_file_name[index], preload=True)
    #     raw_eeg = raw_eeg.get_data()
    #
    #     raw_eeg_channel_var = np.var(raw_eeg, axis=1)
    #     raw_eeg_channel_mean = np.mean(np.abs(raw_eeg), axis=1)
    #
    #     raw_eeg_channel_var_norm = (raw_eeg_channel_var - np.mean(raw_eeg_channel_var)) / np.std(raw_eeg_channel_var)
    #     raw_eeg_channel_mean_norm = (raw_eeg_channel_mean - np.mean(raw_eeg_channel_mean)) / np.std(raw_eeg_channel_mean)
    #
    #     var_bad_channel = list(set(np.where(raw_eeg_channel_var_norm < -1)[0].tolist() + np.where(raw_eeg_channel_var_norm > 1)[0].tolist()))
    #     mean_bad_channel = list(set(np.where(raw_eeg_channel_mean_norm < -1)[0].tolist() + np.where(raw_eeg_channel_mean_norm > 1)[0].tolist()))
    #     channel_list = [i+1 for i in list(set(var_bad_channel + mean_bad_channel))]
    #
    #     channel_list_str = str(channel_list)
    #
    #     f = open(data_dir.segmentation_dir + '\\bad_channel\\' + curren_name_index + '.txt', 'w')
    #     f.write(channel_list_str)
    #     f.close()
    #
    #
    # # eeglab进行第二步处理，ica
    # eng.batch_preprocess2(nargout=0)


    # 根据反应时间等级，画出psd地形图
    num_var = 0  # 记录有多少张地形图
    num_frequence_of_max_power = 0
    num_eeg_power = 0
    num_eeg_power_spectral_entropy = 0
    set_file_name = get_file_name(data_dir.segmentation_dir+r'\preprocess2', '.set')

    for index in range(len(set_file_name)):
        pattern = re.compile('\w+\.set')
        current_name_index = pattern.findall(set_file_name[index])[0][:-4]

        mark, point = read_mark_txt(data_dir.segmentation_dir + r'\raw_data' + '\\' + current_name_index + '.vmrk')

        raw = mne.io.read_raw_eeglab(set_file_name[index], preload=True)

        for i in range(len(mark) - 1):
            raw_eeg = raw.get_data(start=point[i]-1, stop=point[i+1])

            duration = len(raw_eeg[0])
            sliding_window = duration // WINDOWS + 1 # 以0.5秒为一个窗口截取数据计算psd，不足的当作一段
            level_dir = []
            for seg in range(sliding_window):
                if duration < 250:
                    picture_dir = data_dir.features_dir + '\\level1\\{0:0>2}'.format(seg)
                    level_dir.append(picture_dir)
                elif duration > 250 and duration <= 750:
                    picture_dir = data_dir.features_dir + '\\level2\\{0:0>2}'.format(seg)
                    level_dir.append(picture_dir)
                else:
                    picture_dir = data_dir.features_dir + '\\level3\\{0:0>2}'.format(seg)
                    level_dir.append(picture_dir)

                if not os.path.exists(picture_dir): os.makedirs(picture_dir)

            for seg in range(sliding_window):
                start = seg * WINDOWS
                end = (seg + 1) * WINDOWS if ((seg + 1) * WINDOWS) < duration else duration
                if start != end and (end - start) > 200:
                    eeg_four_frequency_band = featureExtract.get_eeg_four_frequency_band(raw_eeg[:, -end:-(start+1)])
                    bands = len(eeg_four_frequency_band)
                    channels = len(eeg_four_frequency_band[0])

                    #——————————————————— 这里可以放置任意多的求特征的函数————————————————————#
                    var = featureExtract.get_eeg_variance(eeg_four_frequency_band)
                    var = var[2, :]

                    frequence_of_max_power = featureExtract.get_frequence_of_max_power(eeg_four_frequency_band)
                    frequence_of_max_power = frequence_of_max_power[2, :]

                    eeg_power = featureExtract.get_eeg_power(eeg_four_frequency_band)
                    eeg_power = eeg_power[2, :]

                    eeg_power_spectral_entropy = featureExtract.get_eeg_power_spectral_entropy(eeg_four_frequency_band)
                    eeg_power_spectral_entropy = eeg_power_spectral_entropy[2, :]
                    # ——————————————————— 这里可以放置任意多的求特征的函数————————————————————#

                    mne.viz.plot_topomap(var, raw.info, show=False)
                    pattern = re.compile('level\d+')
                    lev = pattern.findall(level_dir[seg])[0]
                    var_dir = level_dir[seg] + '\\var'
                    if not os.path.exists(var_dir): os.makedirs(var_dir)
                    plt.savefig(var_dir + '\\{}.jpg'.format(lev + '_' + level_dir[seg][-2:] + '_var' + str(num_var)))
                    plt.close()
                    num_var += 1

                    mne.viz.plot_topomap(frequence_of_max_power, raw.info, show=False)
                    pattern = re.compile('level\d+')
                    lev = pattern.findall(level_dir[seg])[0]
                    frequence_of_max_power_dir = level_dir[seg] + '\\frequence_of_max_power'
                    if not os.path.exists(frequence_of_max_power_dir): os.makedirs(frequence_of_max_power_dir)
                    plt.savefig(frequence_of_max_power_dir + '\\{}.jpg'.format(lev + '_' + level_dir[seg][-2:] + '_frequence_of_max_power' + str(num_frequence_of_max_power)))
                    plt.close()
                    num_frequence_of_max_power += 1

                    mne.viz.plot_topomap(eeg_power, raw.info, show=False)
                    pattern = re.compile('level\d+')
                    lev = pattern.findall(level_dir[seg])[0]
                    eeg_power_dir = level_dir[seg] + '\\eeg_power'
                    if not os.path.exists(eeg_power_dir): os.makedirs(eeg_power_dir)
                    plt.savefig(eeg_power_dir + '\\{}.jpg'.format(lev + '_' + level_dir[seg][-2:] + '_eeg_power' + str(num_eeg_power)))
                    plt.close()
                    num_eeg_power += 1

                    mne.viz.plot_topomap(eeg_power_spectral_entropy, raw.info, show=False)
                    pattern = re.compile('level\d+')
                    lev = pattern.findall(level_dir[seg])[0]
                    eeg_power_spectral_entropy_dir = level_dir[seg] + '\\eeg_power_spectral_entropy'
                    if not os.path.exists(eeg_power_spectral_entropy_dir): os.makedirs(eeg_power_spectral_entropy_dir)
                    plt.savefig(eeg_power_spectral_entropy_dir + '\\{}.jpg'.format(lev + '_' + level_dir[seg][-2:] + '_eeg_power_spectral_entropy' + str(num_eeg_power_spectral_entropy)))
                    plt.close()
                    num_eeg_power_spectral_entropy += 1



