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


def phase_locked_matrix(all_bands_eeg):

    """all_channel_eeg的shape例如是4 * 32 * 8064，其中4是四种频段，32是32个脑电极数脑电极，而8064是每个通道下采集的数据"""

    # 得到输入的频段数，电极通道数和每个通道的采样点数
    bands, channels, points = all_bands_eeg.shape
    eeg_instantaneous_phase = np.zeros_like(all_bands_eeg)  # 初始化每个通道下每个采样点的瞬时相位

    for band, signal_band_eeg in enumerate(all_bands_eeg):
        for channel, single_channel_eeg in enumerate(signal_band_eeg):
            analytic_signal = hilbert(single_channel_eeg)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            eeg_instantaneous_phase[band, channel] = instantaneous_phase

    matrix = np.zeros(shape=[bands, channels, channels])  # 初始化相位锁定矩阵，shape是4 * 32 * 32


    for index in range(bands):
        for i in range(channels):
            for j in range(channels):
                if i == j:
                    matrix[index][i][j] = 1
                else:
                    matrix[index][i][j] = np.abs((compare_elements(eeg_instantaneous_phase[index][i], eeg_instantaneous_phase[index][j])).sum()) / points

        return matrix


if __name__ == '__main__':

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    import data_dir

    eeg = pd.read_csv(data_dir.preprocess_dir + r'\level1\8.csv')
    # print(phase_locked_matrix(eeg.values[:30, 1:]))
    m = phase_locked_matrix(eeg.values[:30, 1:])
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(pd.DataFrame(m),vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True)
    plt.show()
