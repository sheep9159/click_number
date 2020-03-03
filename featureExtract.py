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


def get_eeg_power(eeg_data, Fs=500, win='hamming', scal='spectrum'):

    channels, points = eeg_data.shape
    each_channel_power = []

    for channel in range(channels):
        f, Pper_spec = signal.periodogram(np.array(eeg_data[channel], dtype=float), fs=Fs, window=win, scaling=scal)
        each_channel_power.append(Pper_spec.sum() / len(f))

    return abs(preprocessing.scale(np.array(each_channel_power)))


def get_eeg_power_spectral_entropy(eeg_data, Fs=500, win='hamming', scal='density'):

    channels, points = eeg_data.shape
    each_channel_power_spectral = []
    each_channel_power_spectral_entropy = np.zeros((1, channels))

    for channel in range(channels):
        f, Pper_spec = signal.periodogram(np.array(eeg_data[channel], dtype=float), fs=Fs, window=win, scaling=scal)
        each_channel_power_spectral.append(Pper_spec)

    each_channel_power_spectral = np.array(each_channel_power_spectral).transpose()

    each_channel_power_spectral_entropy = stats.entropy(each_channel_power_spectral)

    return each_channel_power_spectral_entropy


if __name__ == '__main__':

    import data_dir
    import pandas as pd

    eeg = pd.read_csv(data_dir.preprocess_dir + r'\level1\8.csv')
    eeg = eeg.values[:, 1:]
    print(eeg)
    power = get_eeg_power(eeg)
    power_spectral_entropy = get_eeg_power_spectral_entropy(eeg)

    print(power)
    print(power_spectral_entropy)
