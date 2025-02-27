import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import os

def create_white_noise(t, sr = 44100, amp = 0.1):
    noise = np.random.normal(0, amp, int(sr * t))
    return noise

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def draw_oscillogram(signal, main, save=True, save_path=None): 
    plt.figure()
    plt.plot(signal)
    plt.xlabel('Сэмпл')
    plt.ylabel('Колебания')
    plt.title(main)
    plt.grid()
    if save:
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
        else:
            ensure_dir('images')
            plt.savefig(f'images/oscillogram.png')
    plt.show()
    plt.close()

def draw_spectogram(signal, sample_rate, main, save=True, save_path=None):
    plt.figure()
    # Вычисляем спектрограмму вручную
    f, t, Sxx = scipy.signal.spectrogram(signal, fs=sample_rate, nperseg=1024, noverlap=900)
    # Добавляем маленькое значение, чтобы избежать log10(0)
    Sxx = Sxx + 1e-10
    # Преобразуем в децибелы
    Sxx_db = 10 * np.log10(Sxx)
    # Отображаем спектрограмму
    plt.pcolormesh(t, f, Sxx_db, shading='gouraud')
    plt.xlabel('Время, с')
    plt.ylabel('Частота, Гц')
    plt.title(main)
    plt.colorbar(label='Интенсивность, дБ')
    if save:
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
        else:
            ensure_dir('images')
            plt.savefig(f'images/spectogram.png')
    plt.show()
    plt.close()

def draw_amplitude_vs_frequency(signal, sample_rate, save=True, save_path=None):  #TODO: check code
    plt.figure()
    Xf_mag = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(Xf_mag), d=1.0/sample_rate)
    plt.plot(np.abs(freqs), np.abs(Xf_mag))
    plt.title('Амплитудно-частотная характеристика')
    plt.xlabel('Частота, Гц')
    plt.ylabel('Амплитуда, Дб')
    plt.grid()
    if save:
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
        else:
            ensure_dir('images')
            plt.savefig(f'images/amplitude_frequency.png')
    plt.show()
    plt.close()

def highpass_filter(signal, sample_rate, cutoff_freq=1000): 
    order = 5 
    b, a = scipy.signal.butter(order, cutoff_freq / (sample_rate / 2), btype='highpass', analog=False)
    filtered_waveform = scipy.signal.filtfilt(b, a, signal)
    return filtered_waveform

def butterwart_filter(signal, sample_rate, order, cutoff_freq, type='lowpass'): 
    b, a = scipy.signal.butter(order, cutoff_freq / (sample_rate / 2), btype=type, analog=False)
    filtered_waveform = scipy.signal.filtfilt(b, a, signal)
    return filtered_waveform