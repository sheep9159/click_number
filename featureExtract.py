import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from sklearn import preprocessing


def get_eeg_four_frequency_band(eeg_data, Fs=500):
    """
    此函数得到一个多通道eeg信号的四个频段下的数据
    :param eeg_data: 多通道eeg信号数据，如可以是30通道
    :param Fs: eeg信号的采样频率
    :return: 每个通道在四个频段下的滤波后的信号，其形状和输入相同
    """

    channels, points = eeg_data.shape

    theta = [4, 8]
    alpha = [8, 12]
    beta = [12, 30]
    gamma = [30, 42]

    eeg_data_theta = np.zeros_like(eeg_data)
    eeg_data_alpha = np.zeros_like(eeg_data)
    eeg_data_beta = np.zeros_like(eeg_data)
    eeg_data_gamma = np.zeros_like(eeg_data)

    filter_theta = signal.firwin(points, theta, pass_zero='bandpass', fs=Fs)
    filter_alpha = signal.firwin(points, alpha, pass_zero='bandpass', fs=Fs)
    filter_beta = signal.firwin(points, beta, pass_zero='bandpass', fs=Fs)

    '''
        firwin函数设计的都是偶对称的fir滤波器，当N为偶数时，其截止频率处即fs/2都是zero response的，所以用N+1
    '''
    if points % 2 == 0:
        filter_gamma = signal.firwin(points+1, gamma, pass_zero='bandpass', fs=Fs)
    else:
        filter_gamma = signal.firwin(points, gamma, pass_zero='bandpass', fs=Fs)

    for channel in range(channels):
        eeg_data_theta[channel] = signal.convolve(eeg_data[channel], filter_theta, mode='same')
        eeg_data_alpha[channel] = signal.convolve(eeg_data[channel], filter_alpha, mode='same')
        eeg_data_beta[channel] = signal.convolve(eeg_data[channel], filter_beta, mode='same')
        eeg_data_gamma[channel] = signal.convolve(eeg_data[channel], filter_gamma, mode='same')  # 得到fir数字滤波器后直接与信号做卷积

    return np.array([eeg_data_theta, eeg_data_alpha, eeg_data_beta, eeg_data_gamma])  # 将四个频段组合在一起


def get_eeg_variance(all_bands_eeg):
    """
    此函数求得所输入的多频段多通道的eeg数据在每个频段下每个通道的eeg信号的方差，所以输入维度应该是
    [bands, channels, points]三维的，即使数据只有一个频段如α段[channels, points]，也应该
    进行维度扩展，增加一个维度以使得这个函数能正常工作。
    :param all_bands_eeg: 多频段多通道的eeg数据
    :return: 每个频段下每个通道的eeg信号的方差，shape=[bands,channels]
    """
    bands, channels, points = all_bands_eeg.shape
    eeg_variance = np.zeros(shape=[bands, channels])

    for band, signal_band_eeg in enumerate(all_bands_eeg):
        for channel, signal_channel_eeg in enumerate(signal_band_eeg):
            eeg_variance[band, channel] = np.var(signal_channel_eeg)

    return eeg_variance


def get_frequence_of_max_power(all_bands_eeg, Fs=500, win='hamming', scal='spectrum'):
    """
    此函数求得所输入的多频段多通道的eeg数据在每个频段下每个通道的eeg最大能量点所对应的频率，所以输
    入维度应该是[bands, channels, points]三维的，即使数据只有一个频段如α段[channels, points]，
    也应该进行维度扩展，增加一个维度以使得这个函数能正常工作。
    :param all_bands_eeg: 多频段多通道的eeg数据
    :param Fs: 采样频率，默认是500hz
    :param win: 窗函数
    :param scal: 频谱图还是频谱密度图，可选density
    :return: 每个频段下每个通道的eeg最大能量点所对应的频率，shape=[bands，channels]
    """
    bands, channels, points = all_bands_eeg.shape
    frequence_of_max_power = np.zeros(shape=[bands, channels])
    for band, signal_band_eeg in enumerate(all_bands_eeg):
        for channel, signal_channel_eeg in enumerate(signal_band_eeg):
            f, Pper_spec = signal.periodogram(np.array(signal_channel_eeg, dtype=float), fs=Fs, window=win, scaling=scal)
            frequence_of_max_power[band, channel] = f[np.argmax(Pper_spec)]

    return frequence_of_max_power


def get_eeg_power(all_bands_eeg, Fs=500, win='hamming', scal='spectrum'):
    """
    此函数求得所输入的多频段多通道的eeg数据在每个频段下每个通道的eeg能量值，所以输入维度应该是
    [bands, channels, points]三维的，即使数据只有一个频段如α段[channels, points]，也应该
    进行维度扩展，增加一个维度以使得这个函数能正常工作。
    :param all_bands_eeg: 多频段多通道的eeg数据
    :param Fs: 采样频率，默认是500hz
    :param win: 窗函数
    :param scal: 频谱图还是频谱密度图，可选density
    :return: 每个频段下每个通道的eeg信号能量（被均一化了的），shape=[bands，channels]
    """
    bands, channels, points = all_bands_eeg.shape
    eeg_power = np.zeros(shape=[bands, channels])

    for band, signal_band_eeg in enumerate(all_bands_eeg):
        for channel, signal_channel_eeg in enumerate(signal_band_eeg):
            f, Pper_spec = signal.periodogram(np.array(signal_channel_eeg, dtype=float), fs=Fs, window=win, scaling=scal)
            signal_channel_eeg_power = Pper_spec.sum() / len(f)
            eeg_power[band, channel] = np.array(signal_channel_eeg_power)

    return preprocessing.scale(eeg_power, axis=1)


def get_eeg_power_spectral_entropy(all_bands_eeg, Fs=500, win='hamming', scal='density'):
    """
    此函数求得所输入的多频段多通道的eeg数据在每个频段下每个通道的eeg功率谱熵，所以输入维度应该是
    [bands, channels, points]三维的即使数据只有一个频段如α段[channels, points]，也应该
    进行维度扩展，增加一个维度以使得这个函数能正常工作。
    :param all_bands_eeg: 多频段多通道的eeg数据
    :param Fs: 采样频率，默认是500hz
    :param win: 窗函数
    :param scal: 频谱图还是频谱密度图，可选density
    :return: 每个频段下每个通道的eeg的功率谱熵，shape=[bands,channels]
    """
    bands, channels, points = all_bands_eeg.shape

    eeg_power_spectral_entropy = np.zeros(shape=[bands, channels])

    # 这里(points//2+1)是因为一个长度为N的信号，求得其频谱图得到的数据点只有N//2+1个
    signal_band_power_spectral = np.zeros(shape=[channels, (points//2+1)])

    for band, signal_band_eeg in enumerate(all_bands_eeg):
        for channel, signal_channel_eeg in enumerate(signal_band_eeg):
            f, Pper_spec = signal.periodogram(np.array(signal_channel_eeg, dtype=float), fs=Fs, window=win, scaling=scal)
            signal_band_power_spectral[channel, :] = Pper_spec

        each_channel_power_spectral = signal_band_power_spectral.transpose()

        eeg_power_spectral_entropy[band, :] = stats.entropy(each_channel_power_spectral)

    return eeg_power_spectral_entropy


if __name__ == '__main__':

    import data_dir

    import os
    import re
    import pandas as pd
    import matplotlib.pyplot as plt

    index = ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F8', 'F3', 'Fz', 'F4', 'FC5', 'FC6', 'T7', 'T8', 'C3',
             'Cz', 'C4', 'CP5', 'CP6', 'P7', 'P8', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'PO3', 'PO4', 'O1',
             'O2']

    num_electrodes = 29


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

    preprocessing_eeg_file = get_file_name(data_dir.preprocess_dir, '.csv')

    for file in preprocessing_eeg_file:

        level = re.findall('level\d+', file)[0]
        name = re.findall('\d+', file)[-1]

        print(level, ',  ', name)

        eeg = pd.read_csv(file)
        eeg = eeg.values[:, 1:]

        four_band_eeg = get_eeg_four_frequency_band(eeg)

        variance = get_eeg_variance(four_band_eeg)
        frequence_of_max_power = get_frequence_of_max_power(four_band_eeg)
        power = get_eeg_power(four_band_eeg)
        power_spectral_entropy = get_eeg_power_spectral_entropy(four_band_eeg)

        fig = plt.figure(figsize=(25, 10))

        #  画eeg_variance
        plt.subplot(221)
        plt.plot(np.arange(num_electrodes), variance[0], 'o-', color='blue', label='theta')
        plt.plot(np.arange(num_electrodes), variance[1], 'o-', color='cyan', label='alpha')
        plt.plot(np.arange(num_electrodes), variance[2], 'o-', color='brown', label='beta')
        plt.plot(np.arange(num_electrodes), variance[3], 'o-', color='black', label='gamma')
        plt.legend(loc='upper right')
        plt.xticks(np.arange(num_electrodes), index)
        plt.xlabel('electrodes')
        plt.title('eeg_variance', color='red')

        #  画frequence_of_max_power
        plt.subplot(222)
        plt.plot(np.arange(num_electrodes), frequence_of_max_power[0], 'o-', color='blue', label='theta')
        plt.plot(np.arange(num_electrodes), frequence_of_max_power[1], 'o-', color='cyan', label='alpha')
        plt.plot(np.arange(num_electrodes), frequence_of_max_power[2], 'o-', color='brown', label='beta')
        plt.plot(np.arange(num_electrodes), frequence_of_max_power[3], 'o-', color='black', label='gamma')
        plt.legend(loc='upper right')
        plt.xticks(np.arange(num_electrodes), index)
        plt.xlabel('electrodes')
        plt.title('frequence_of_max_power', color='red')

        #  画power
        plt.subplot(223)
        plt.plot(np.arange(num_electrodes), power[0], 'o-', color='blue', label='theta')
        plt.plot(np.arange(num_electrodes), power[1], 'o-', color='cyan', label='alpha')
        plt.plot(np.arange(num_electrodes), power[2], 'o-', color='brown', label='beta')
        plt.plot(np.arange(num_electrodes), power[3], 'o-', color='black', label='gamma')
        plt.legend(loc='upper right')
        plt.xticks(np.arange(num_electrodes), index)
        plt.xlabel('electrodes')
        plt.title('power', color='red')

        #  画power_spectral_entropy
        plt.subplot(224)
        plt.plot(np.arange(num_electrodes), power_spectral_entropy[0], 'o-', color='blue', label='theta')
        plt.plot(np.arange(num_electrodes), power_spectral_entropy[1], 'o-', color='cyan', label='alpha')
        plt.plot(np.arange(num_electrodes), power_spectral_entropy[2], 'o-', color='brown', label='beta')
        plt.plot(np.arange(num_electrodes), power_spectral_entropy[3], 'o-', color='black', label='gamma')
        plt.legend(loc='upper right')
        plt.xticks(np.arange(num_electrodes), index)
        plt.xlabel('electrodes')
        plt.title('power_spectral_entropy', color='red')

        plt.subplots_adjust(wspace=0.3, hspace=0.3)  # 调整子图间距


        if level == 'level1':plt.savefig(data_dir.features_dir + r'\level1\{}.png'.format(name), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
        elif level == 'level2':plt.savefig(data_dir.features_dir + r'\level2\{}.png'.format(name), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
        elif level == 'level3':plt.savefig(data_dir.features_dir + r'\level3\{}.png'.format(name), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)

        plt.close()
