import torchaudio
import matplotlib.pyplot as plt
import os
import abc
from scripts import (draw_oscillogram, draw_spectogram, draw_amplitude_vs_frequency, 
                    create_white_noise, ensure_dir, highpass_filter, butterwart_filter)
import scipy.signal
import numpy as np


class AudioStream(abc.ABC):
    """
    Абстрактный класс для работы с различными аудиопотоками.
    Определяет общий интерфейс для всех типов аудиоданных.
    """
    def __init__(self, name):
        """
        Инициализация аудиопотока.
        
        Args:
            name (str): Имя аудиопотока для создания папки с изображениями
        """
        self.name = name
        self.images_dir = f"images/{self.name}"
        ensure_dir(self.images_dir)
        
        self.mono_dir = f"{self.images_dir}/mono"
        self.stereo_dir = f"{self.images_dir}/stereo"
    
    @abc.abstractmethod
    def load(self):
        """
        Загрузка аудиоданных.
        
        Returns:
            tuple: (signal, sample_rate) - аудиосигнал и частота дискретизации
        """
        pass
    
    def trim_audio(self, signal, sample_rate, max_duration=20):
        """
        Обрезает аудиосигнал до указанной максимальной длительности.
        
        Args:
            signal: Аудиосигнал
            sample_rate: Частота дискретизации
            max_duration (float): Максимальная длительность в секундах
            
        Returns:
            tensor: Обрезанный аудиосигнал
        """
        duration = signal.shape[-1] / sample_rate
        
        if duration > max_duration:
            max_samples = int(max_duration * sample_rate)
            
            trimmed_signal = signal[..., :max_samples]
            
            print(f"Аудио обрезано с {duration:.2f}с до {max_duration:.2f}с")
            return trimmed_signal
        
        return signal
    
    def save_image(self, image_name, processing_type="default"):
        """
        Возвращает полный путь для сохранения изображения.
        
        Args:
            image_name (str): Имя изображения без расширения
            processing_type (str): Тип обработки ('default', 'mono', 'stereo')
            
        Returns:
            str: Полный путь для сохранения изображения
        """
        if processing_type == "mono":
            ensure_dir(self.mono_dir)
            return f"{self.mono_dir}/{image_name}.png"
        elif processing_type == "stereo":
            ensure_dir(self.stereo_dir)
            return f"{self.stereo_dir}/{image_name}.png"
        else:
            return f"{self.images_dir}/{image_name}.png"
    
    def analyze(self, signal, sample_rate, prefix="", processing_type="default"):
        """
        Проводит анализ сигнала и сохраняет результаты.
        
        Args:
            signal: Аудиосигнал
            sample_rate: Частота дискретизации
            prefix (str): Префикс для имен файлов
            processing_type (str): Тип обработки ('default', 'mono', 'stereo')
        """
        prefix = f"{prefix}_" if prefix else ""
        
        draw_oscillogram(signal, f"Осциллограмма {self.name}", save=True, 
                         save_path=self.save_image(f"{prefix}oscillogram", processing_type))
        draw_spectogram(signal, sample_rate, f"Спектрограмма {self.name}", save=True, 
                       save_path=self.save_image(f"{prefix}spectogram", processing_type))
        draw_amplitude_vs_frequency(signal, sample_rate, save=True, 
                                   save_path=self.save_image(f"{prefix}amplitude_frequency", processing_type))
    
    def convert_to_mono(self, signal):
        """
        Преобразует стерео сигнал в моно путем усреднения каналов.
        
        Args:
            signal: Стерео сигнал (тензор формы [channels, samples])
            
        Returns:
            tensor: Моно сигнал (тензор формы [1, samples])
        """
        import torch
        
        if signal.dim() == 1:
            return signal.unsqueeze(0)
        
        mono_signal = torch.mean(signal, dim=0, keepdim=True)
        return mono_signal
    
    def process_stereo_channels(self, signal, sample_rate, prefix=""):
        """
        Обрабатывает каналы стерео сигнала и отображает их на одном изображении.
        
        Args:
            signal: Стерео сигнал (тензор формы [channels, samples])
            sample_rate: Частота дискретизации
            prefix (str): Префикс для имен файлов
            
        Returns:
            dict: Словарь с обработанными каналами
        """
        import torch
        
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        
        num_channels = signal.shape[0]
        channels = {}
        
        if num_channels > 1:
            plt.figure(figsize=(12, 4 * num_channels))
            for i in range(num_channels):
                channel_signal = signal[i]
                channels[f"channel_{i+1}"] = channel_signal
                
                plt.subplot(num_channels, 1, i+1)
                plt.plot(channel_signal)
                plt.title(f"Канал {i+1} - Осциллограмма")
                plt.xlabel('Сэмпл')
                plt.ylabel('Колебания')
                plt.grid(True)
            
            plt.tight_layout()
            save_path = self.save_image(f"{prefix}_all_channels_oscillogram", "stereo")
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
            plt.close()
            
            plt.figure(figsize=(12, 4 * num_channels))
            for i in range(num_channels):
                channel_signal = signal[i]
                
                plt.subplot(num_channels, 1, i+1)
                f, t, Sxx = scipy.signal.spectrogram(channel_signal, fs=sample_rate, nperseg=1024, noverlap=900)
                Sxx = Sxx + 1e-10
                Sxx_db = 10 * np.log10(Sxx)
                plt.pcolormesh(t, f, Sxx_db, shading='gouraud')
                plt.title(f"Канал {i+1} - Спектрограмма")
                plt.xlabel('Время, с')
                plt.ylabel('Частота, Гц')
                plt.colorbar(label='Интенсивность, дБ')
            
            plt.tight_layout()
            save_path = self.save_image(f"{prefix}_all_channels_spectogram", "stereo")
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
            plt.close()
            
            plt.figure(figsize=(12, 4 * num_channels))
            for i in range(num_channels):
                channel_signal = signal[i]
                
                plt.subplot(num_channels, 1, i+1)
                Xf_mag = np.fft.fft(channel_signal)
                freqs = np.fft.fftfreq(len(Xf_mag), d=1.0/sample_rate)
                plt.plot(np.abs(freqs), np.abs(Xf_mag))
                plt.title(f"Канал {i+1} - Амплитудно-частотная характеристика")
                plt.xlabel('Частота, Гц')
                plt.ylabel('Амплитуда, Дб')
                plt.grid(True)
            
            plt.tight_layout()
            save_path = self.save_image(f"{prefix}_all_channels_amplitude_frequency", "stereo")
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
            plt.close()
            
            print(f"Создано объединенное изображение для {num_channels} каналов")
        else:
            channel_signal = signal[0]
            channels["channel_1"] = channel_signal
            self.analyze(channel_signal, sample_rate, 
                         prefix=f"{prefix}_channel_1" if prefix else "channel_1", 
                         processing_type="stereo")
        
        return channels
    
    def apply_filters(self, signal, sample_rate, processing_type="default"):
        """
        Применяет различные фильтры к сигналу и сохраняет результаты.
        
        Args:
            signal: Аудиосигнал
            sample_rate: Частота дискретизации
            processing_type (str): Тип обработки ('default', 'mono', 'stereo')
            
        Returns:
            dict: Словарь с отфильтрованными сигналами
        """
        filtered_signals = {}
        
        highpass_filtered = highpass_filter(signal, sample_rate, cutoff_freq=1000)
        filtered_signals['filter1_highpass'] = highpass_filtered
        self.analyze(highpass_filtered, sample_rate, prefix="filter1_highpass", processing_type=processing_type)
        print(f"Применен фильтр верхних частот (№1) с частотой среза 1000 Гц")
        
        butterworth_filtered = butterwart_filter(signal, sample_rate, order=5, cutoff_freq=2000, type='lowpass')
        filtered_signals['filter9_butterworth'] = butterworth_filtered
        self.analyze(butterworth_filtered, sample_rate, prefix="filter9_butterworth", processing_type=processing_type)
        print(f"Применен фильтр Баттерворта (№9) с частотой среза 2000 Гц")
        
        combined_filtered = butterwart_filter(highpass_filtered, sample_rate, order=5, cutoff_freq=2000, type='lowpass')
        filtered_signals['combined_filters'] = combined_filtered
        self.analyze(combined_filtered, sample_rate, prefix="combined_filters", processing_type=processing_type)
        print(f"Применены последовательно оба фильтра: верхних частот и Баттерворта")
        
        return filtered_signals

    def apply_filters_to_stereo(self, signal, sample_rate, prefix=""):
        """
        Применяет фильтры к стерео сигналу и отображает результаты на одном изображении.
        
        Args:
            signal: Стерео сигнал (тензор формы [channels, samples])
            sample_rate: Частота дискретизации
            prefix (str): Префикс для имен файлов
            
        Returns:
            dict: Словарь с отфильтрованными сигналами
        """
        import torch
        
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        
        num_channels = signal.shape[0]
        filtered_channels = {}
        
        if num_channels > 1:
            plt.figure(figsize=(12, 4 * num_channels))
            highpass_filtered = {}
            
            for i in range(num_channels):
                channel_signal = signal[i]
                filtered = highpass_filter(channel_signal, sample_rate, cutoff_freq=1000)
                highpass_filtered[f"channel_{i+1}"] = filtered
                
                plt.subplot(num_channels, 1, i+1)
                plt.plot(filtered)
                plt.title(f"Канал {i+1} - Фильтр верхних частот (№1)")
                plt.xlabel('Сэмпл')
                plt.ylabel('Колебания')
                plt.grid(True)
            
            plt.tight_layout()
            save_path = self.save_image(f"{prefix}_all_channels_filter1_highpass", "stereo")
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
            plt.close()
            
            filtered_channels['filter1_highpass'] = highpass_filtered
            
            plt.figure(figsize=(12, 4 * num_channels))
            butterworth_filtered = {}
            
            for i in range(num_channels):
                channel_signal = signal[i]
                filtered = butterwart_filter(channel_signal, sample_rate, order=5, cutoff_freq=2000, type='lowpass')
                butterworth_filtered[f"channel_{i+1}"] = filtered
                
                plt.subplot(num_channels, 1, i+1)
                plt.plot(filtered)
                plt.title(f"Канал {i+1} - Фильтр Баттерворта (№9)")
                plt.xlabel('Сэмпл')
                plt.ylabel('Колебания')
                plt.grid(True)
            
            plt.tight_layout()
            save_path = self.save_image(f"{prefix}_all_channels_filter9_butterworth", "stereo")
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
            plt.close()
            
            filtered_channels['filter9_butterworth'] = butterworth_filtered
            
            plt.figure(figsize=(12, 4 * num_channels))
            combined_filtered = {}
            
            for i in range(num_channels):
                filtered = butterwart_filter(highpass_filtered[f"channel_{i+1}"], sample_rate, order=5, cutoff_freq=2000, type='lowpass')
                combined_filtered[f"channel_{i+1}"] = filtered
                
                plt.subplot(num_channels, 1, i+1)
                plt.plot(filtered)
                plt.title(f"Канал {i+1} - Последовательное применение фильтров")
                plt.xlabel('Сэмпл')
                plt.ylabel('Колебания')
                plt.grid(True)
            
            plt.tight_layout()
            save_path = self.save_image(f"{prefix}_all_channels_combined_filters", "stereo")
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
            plt.close()
            
            filtered_channels['combined_filters'] = combined_filtered
            
            print(f"Применены фильтры к {num_channels} каналам и созданы объединенные изображения")
        else:
            filtered_channels = self.apply_filters(signal[0], sample_rate, processing_type="stereo")
        
        return filtered_channels


class Audio(AudioStream):
    """
    Класс для работы с аудиофайлами.
    """
    def __init__(self, audio_file: str):
        self.audio_file = audio_file
        file_name = os.path.splitext(os.path.basename(audio_file))[0]
        super().__init__(file_name)

    def load(self, max_duration=20):
        """
        Загружает аудиофайл и обрезает его, если длительность превышает max_duration.
        
        Args:
            max_duration (float): Максимальная длительность в секундах
            
        Returns:
            tuple: (signal, sample_rate) - аудиосигнал и частота дискретизации
        """
        signal, sample_rate = torchaudio.load(uri=self.audio_file)
        
        duration = signal.shape[1] / sample_rate
        print(f"Частота дискретизации: {sample_rate} Гц, torchaudio сохраняет исходную частоту дискретизации")
        print(f"Количество каналов: {signal.shape[0]}")
        print(f"Длина сигнала: {signal.shape[1]}")
        print(f"Тип сигнала: {signal.dtype}")
        print(f"Диапазон значений сигнала: {signal.min()} до {signal.max()}")
        print(f"Длительность сигнала: {duration:.2f}с")
        
        signal = self.trim_audio(signal, sample_rate, max_duration)
        
        return signal, sample_rate
    
    def process_audio(self):
        """
        Обрабатывает аудиофайл разными способами: исходный сигнал, моно и по каналам.
        """
        signal, sample_rate = self.load()
        
        num_channels = signal.shape[0]
        
        if num_channels == 1:
            print(f"\nОбработка одноканального сигнала:")
            self.analyze(signal[0], sample_rate)
            filtered_signals = self.apply_filters(signal[0], sample_rate)
            
            return {
                'original': filtered_signals
            }
        else:
            print(f"\nОбработка моно сигнала (усреднение каналов):")
            mono_signal = self.convert_to_mono(signal)
            self.analyze(mono_signal[0], sample_rate, processing_type="mono")
            filtered_mono = self.apply_filters(mono_signal[0], sample_rate, processing_type="mono")
            
            print(f"\nОбработка каналов стерео сигнала:")
            channels = self.process_stereo_channels(signal, sample_rate)
            
            filtered_stereo = self.apply_filters_to_stereo(signal, sample_rate)
            
            return {
                'mono': filtered_mono,
                'stereo': filtered_stereo
            }


class WhiteNoise(AudioStream):
    """
    Класс для работы с белым шумом.
    """
    def __init__(self, duration, sample_rate=44100, amplitude=0.1):
        self.duration = min(duration, 20)
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        name = f"white_noise_{self.duration}s_{sample_rate}hz"
        super().__init__(name)
        self.noise = None
    
    def load(self):
        self.noise = create_white_noise(self.duration, self.sample_rate, self.amplitude)
        print(f"Создан белый шум длительностью {self.duration}с")
        print(f"Частота дискретизации: {self.sample_rate} Гц")
        print(f"Амплитуда шума: {self.amplitude}")
        print(f"Длина сигнала: {len(self.noise)}")
        print(f"Диапазон значений сигнала: {self.noise.min()} до {self.noise.max()}")
        return self.noise, self.sample_rate


if __name__ == "__main__":
    plt.ion()
    
    audio = Audio("data/00001.wav")
    results = audio.process_audio()
    
    noise = WhiteNoise(6, 44100)
    noise_signal, noise_sr = noise.load()
    noise.analyze(noise_signal, noise_sr)
    filtered_noise = noise.apply_filters(noise_signal, noise_sr)
    
    audio2 = Audio("data/аннигиляторная пушка.wav")
    results2 = audio2.process_audio()
    
    audio3 = Audio("data/organ.mp3")
    results3 = audio3.process_audio()
    
    plt.ioff()
    plt.show(block=True)

