import data_dir
import segmentation
import featureExtract

import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matlab.engine
import re
import mne


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


eng = matlab.engine.start_matlab()

# 从原数据中分段
mark_file_name = get_file_name(data_dir.raw_eeg_dir, '.vmrk')
vhdr_file_name = get_file_name(data_dir.raw_eeg_dir, '.vhdr')
eeg_file_name = get_file_name(data_dir.raw_eeg_dir, '.eeg')

for i in range(len(eeg_file_name)):

    mark, point_boundary, name_index, trail_indices, datas_length = segmentation.read_mark_txt(mark_file_name[i])
    segmentation.read_vhdr_file(vhdr_file_name[i], mark, name_index, trail_indices, datas_length)
    segmentation.read_eeg_file(eeg_file_name[i], mark, point_boundary, name_index, trail_indices)


# eeglab进行第一步处理
eng.batch_preprocess1(nargout=0)


# # 找到bad_channel
# set_file_name = get_file_name(data_dir.segmentation_dir+r'\preprocess1', '.set')
#
# for index in range(len(set_file_name)):
#     pattern = re.compile('\w+\.set')
#     name_index = pattern.findall(set_file_name[index])[0][:-4]
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
#     var_bad_channel = list(set(np.where(raw_eeg_channel_var_norm < -2)[0].tolist() + np.where(raw_eeg_channel_var_norm > 2)[0].tolist()))
#     mean_bad_channel = list(set(np.where(raw_eeg_channel_mean_norm < -2)[0].tolist() + np.where(raw_eeg_channel_mean_norm > 2)[0].tolist()))
#     channel_list = [i+1 for i in list(set(var_bad_channel + mean_bad_channel))]
#
#     channel_list_str = str(channel_list)
#
#     f = open(data_dir.segmentation_dir+'\\bad_channel\\'+name_index+'.txt', 'w')
#     f.write(channel_list_str)
#     f.close()
#
#
# # eeglab进行第二步处理
# eng.batch_preprocess2(nargout=0)
#
#
# # 根据反应时间等级，画出psd地形图
# set_file_name = get_file_name(data_dir.segmentation_dir+r'\preprocess2', '.set')
#
# for index in range(len(set_file_name)):
#     pattern = re.compile('\w+\.set')
#     name_index = pattern.findall(set_file_name[index])[0][:-4]
#
#     raw = mne.io.read_raw_eeglab(set_file_name[index], preload=True)
#     raw_eeg = raw.get_data()
#
#     duration = len(raw_eeg)
#     sliding_window = duration // 250 + 1 # 以0.5秒为一个窗口截取数据计算psd，不足的当作一段
#     level_dir = []
#     for i in range(sliding_window):
#         if duration < 250:
#             level_dir.append(data_dir.features_dir + '\\level1\\{}'.format(i))
#         elif duration > 250 and duration <= 750:
#             level_dir.append(data_dir.features_dir + '\\level2\\{}'.format(i))
#         else:
#             level_dir.append(data_dir.features_dir + '\\level3\\{}'.format(i))
#
#         if not os.path.exists(level_dir[i]): os.makedirs(level_dir[i])
#
#     for seg in range(sliding_window):
#         start = seg * 250
#         end = (seg + 1) * 250 if ((seg + 1) * 250) < duration else duration
#         eeg_four_frequency_band = featureExtract.get_eeg_four_frequency_band(raw_eeg[start:end])
#         bands = len(eeg_four_frequency_band)
#         channels = len(eeg_four_frequency_band[0])
#
#         psd = np.zeros(shape=[bands, channels])
#
#         for band in range(bands):
#             for channel in range(channels):
#                 f, Pper_spec = signal.periodogram(eeg_four_frequency_band[band, channel], 500, 'hamming', scaling='spectrum')
#                 psd[band, channel] = Pper_spec.sum()
#
#         psd = psd / psd.sum(axis=1).reshape(bands,-1)
#         psd = psd[2,:]
#         mne.viz.plot_topomap(psd, raw.info, show=False)
#         plt.savefig(level_dir[seg] + '\\{},jpg'.format(name_index))


