import numpy as np
import os
import mne
import data_dir
import re


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


if __name__ == '__main__':
    eeg_file_name = get_file_name(data_dir.segmentation_dir+r'\preprocess1', '.set')

    for index in range(len(eeg_file_name)):
        pattern = re.compile('\w+\.set')
        name_index = pattern.findall(eeg_file_name[index])[0][:-4]

        raw_eeg = mne.io.read_raw_eeglab(eeg_file_name[index], preload=True)
        raw_eeg = raw_eeg.get_data()

        raw_eeg_channel_var = np.var(raw_eeg, axis=1)
        raw_eeg_channel_mean = np.mean(np.abs(raw_eeg), axis=1)

        raw_eeg_channel_var_norm = (raw_eeg_channel_var - np.mean(raw_eeg_channel_var)) / np.std(raw_eeg_channel_var)
        raw_eeg_channel_mean_norm = (raw_eeg_channel_mean - np.mean(raw_eeg_channel_mean)) / np.std(raw_eeg_channel_mean)

        var_bad_channel = list(set(np.where(raw_eeg_channel_var_norm > 0.7)[0].tolist() + np.where(raw_eeg_channel_var_norm < -0.7)[0].tolist()))
        mean_bad_channel = list(set(np.where(raw_eeg_channel_mean_norm < -0.7)[0].tolist() + np.where(raw_eeg_channel_mean_norm > 0.7)[0].tolist()))
        channel_list = [i+1 for i in list(set(var_bad_channel + mean_bad_channel))]

        channel_list_str = str(channel_list)

        f = open(data_dir.segmentation_dir+'\\bad_channel\\'+name_index+'.txt', 'w')
        f.write(channel_list_str)
        f.close()
