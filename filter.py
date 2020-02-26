import numpy as np
import scipy.signal as signal


def get_eeg_signal_four_frequency_band(eeg_signal_data, Fs=500):

    channels, points = eeg_signal_data.shape

    theta = [4, 8]
    alpha = [8, 12]
    beta = [12, 30]
    gamma = 30

    eeg_data_theta = np.zeros_like(eeg_signal_data)
    eeg_data_alpha = np.zeros_like(eeg_signal_data)
    eeg_data_beta = np.zeros_like(eeg_signal_data)
    eeg_data_gamma = np.zeros_like(eeg_signal_data)

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
        eeg_data_theta[channel] = signal.convolve(eeg_signal_data[channel], filter_theta, mode='same')
        eeg_data_alpha[channel] = signal.convolve(eeg_signal_data[channel], filter_alpha, mode='same')
        eeg_data_beta[channel] = signal.convolve(eeg_signal_data[channel], filter_beta, mode='same')
        eeg_data_gamma[channel] = signal.convolve(eeg_signal_data[channel], filter_gamma, mode='same')  # 得到fir数字滤波器后直接与信号做卷积

    return np.array([eeg_data_theta, eeg_data_alpha, eeg_data_beta, eeg_data_gamma])  # 将四个频段组合在一起