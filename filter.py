import numpy as np
import scipy.signal as signal
from sklearn import preprocessing


def get_eeg_four_frequency_band(eeg_data, Fs=500):

    channels, points = eeg_data.shape

    theta = [4, 8]
    alpha = [8, 12]
    beta = [12, 30]
    gamma = 30

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
        filter_gamma = signal.firwin(points+1, gamma, pass_zero='highpass', fs=128)
    else:
        filter_gamma = signal.firwin(points, gamma, pass_zero='highpass', fs=128)

    for channel in range(channels):
        eeg_data_theta[channel] = signal.convolve(eeg_data[channel], filter_theta, mode='same')
        eeg_data_alpha[channel] = signal.convolve(eeg_data[channel], filter_alpha, mode='same')
        eeg_data_beta[channel] = signal.convolve(eeg_data[channel], filter_beta, mode='same')
        eeg_data_gamma[channel] = signal.convolve(eeg_data[channel], filter_gamma, mode='same')  # 得到fir数字滤波器后直接与信号做卷积

    return np.array([eeg_data_theta, eeg_data_alpha, eeg_data_beta, eeg_data_gamma])  # 将四个频段组合在一起

def get_eeg_is_useful_from_psd(eeg_data, Fs=500, win='hamming', scal='spectrum', threshold=0.5):
    channels, points = eeg_data.shape
    eeg_data = preprocessing.scale(eeg_data)
    each_channel_high_frequency_percent = np.zeros(channels)
    for channel in range(channels):
        f, Pper_spec = signal.periodogram(eeg_data[channel], fs=Fs, window=win, scaling=scal)
        spec = Pper_spec.sum()
        high_spec = Pper_spec[11:].sum()
        each_channel_high_frequency_percent[channel] = high_spec / spec

    high_frequency_percent = each_channel_high_frequency_percent.sum() / channels
    if high_frequency_percent > threshold:
        return False
    else:
        return True

