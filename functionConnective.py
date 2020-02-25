from scipy.signal import hilbert
import numpy as np
import matplotlib.pyplot as plt


def compare_elements(array1, array2):  # array1和array2大小相同

    array = np.zeros(len(array1))
    for i in range(len(array1)):
        if array1[i] == array2[i]:
            array[i] = 0
        elif array1[i] > array2[i]:
            array[i] = 1
        else:
            array[i] = -1

    return array


def phase_locked_matrix(all_channel_eeg):

    """all_channel_eeg的shape例如是32 * 8064，其中32是脑电极，而8064是每个通道下采集的数据"""

    channels = all_channel_eeg.shape[0]
    sampling_number = all_channel_eeg.shape[1]  # 得到输入的通道数和每个通道的采样点数
    eeg_instantaneous_phase = np.zeros_like(all_channel_eeg)  # 初始化每个通道下每个采样点的瞬时相位


    for index, single_channel_eeg in enumerate(all_channel_eeg):
        analytic_signal = hilbert(single_channel_eeg)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        eeg_instantaneous_phase[index] = instantaneous_phase

    matrix = np.zeros(shape=[channels, channels])  # 初始化相位锁定矩阵，shape是32 * 32


    for i in range(channels):
        for j in range(channels):
            if i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.abs((compare_elements(eeg_instantaneous_phase[i], eeg_instantaneous_phase[j])).sum()) / sampling_number

    return matrix


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    eeg = pd.read_csv(r'D:\Files\SJTU\Study\MME_Lab\Teacher_Lu\click_number\eeg\process1.0\fast\8.csv')
    # print(phase_locked_matrix(eeg.values[:30, 1:]))
    m = phase_locked_matrix(eeg.values[:30, 1:])
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(pd.DataFrame(m),vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True)
    plt.show()
