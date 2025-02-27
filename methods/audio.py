import torchaudio
import matplotlib.pyplot as plt
import os
import abc
from scripts import (draw_oscillogram, draw_spectogram, draw_amplitude_vs_frequency, 
                    create_white_noise, ensure_dir, highpass_filter, butterwart_filter)


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
        # Создаем путь к папке для изображений
        self.images_dir = f"images/{self.name}"
        # Создаем папку, если она не существует
        ensure_dir(self.images_dir)
    
    @abc.abstractmethod
    def load(self):
        """
        Загрузка аудиоданных.
        
        Returns:
            tuple: (signal, sample_rate) - аудиосигнал и частота дискретизации
        """
        pass
    
    def save_image(self, image_name):
        """
        Возвращает полный путь для сохранения изображения.
        
        Args:
            image_name (str): Имя изображения без расширения
            
        Returns:
            str: Полный путь для сохранения изображения
        """
        return f"{self.images_dir}/{image_name}.png"
    
    def analyze(self, signal, sample_rate, prefix=""):
        """
        Проводит анализ сигнала и сохраняет результаты.
        
        Args:
            signal: Аудиосигнал
            sample_rate: Частота дискретизации
            prefix (str): Префикс для имен файлов
        """
        # Добавляем подчеркивание к префиксу, если он не пустой
        prefix = f"{prefix}_" if prefix else ""
        
        draw_oscillogram(signal, f"Осциллограмма {self.name}", save=True, 
                         save_path=self.save_image(f"{prefix}oscillogram"))
        draw_spectogram(signal, sample_rate, f"Спектрограмма {self.name}", save=True, 
                       save_path=self.save_image(f"{prefix}spectogram"))
        draw_amplitude_vs_frequency(signal, sample_rate, save=True, 
                                   save_path=self.save_image(f"{prefix}amplitude_frequency"))
    
    def apply_filters(self, signal, sample_rate):
        """
        Применяет различные фильтры к сигналу и сохраняет результаты.
        
        Args:
            signal: Аудиосигнал
            sample_rate: Частота дискретизации
            
        Returns:
            dict: Словарь с отфильтрованными сигналами
        """
        filtered_signals = {}
        
        # Фильтр №1 - Фильтр верхних частот
        highpass_filtered = highpass_filter(signal, sample_rate, cutoff_freq=1000)
        filtered_signals['filter1_highpass'] = highpass_filtered
        self.analyze(highpass_filtered, sample_rate, prefix="filter1_highpass")
        print(f"Применен фильтр верхних частот (№1) с частотой среза 1000 Гц")
        
        # Фильтр №9 - Фильтр Баттерворта
        butterworth_filtered = butterwart_filter(signal, sample_rate, order=5, cutoff_freq=2000, type='lowpass')
        filtered_signals['filter9_butterworth'] = butterworth_filtered
        self.analyze(butterworth_filtered, sample_rate, prefix="filter9_butterworth")
        print(f"Применен фильтр Баттерворта (№9) с частотой среза 2000 Гц")
        
        # Последовательное применение обоих фильтров
        combined_filtered = butterwart_filter(highpass_filtered, sample_rate, order=5, cutoff_freq=2000, type='lowpass')
        filtered_signals['combined_filters'] = combined_filtered
        self.analyze(combined_filtered, sample_rate, prefix="combined_filters")
        print(f"Применены последовательно оба фильтра: верхних частот и Баттерворта")
        
        return filtered_signals


class Audio(AudioStream):
    """
    Класс для работы с аудиофайлами.
    """
    def __init__(self, audio_file: str):
        self.audio_file = audio_file
        file_name = os.path.splitext(os.path.basename(audio_file))[0]
        super().__init__(file_name)

    def load(self):
        signal, sample_rate = torchaudio.load(uri=self.audio_file)
        print(f"Частота дискретизации: {sample_rate} Гц, torchaudio сохраняет исходную частоту дискретизации")
        print(f"Количество каналов: {signal.shape[0]}")
        print(f"Длина сигнала: {signal.shape[1]}")
        print(f"Тип сигнала: {signal.dtype}")
        print(f"Диапазон значений сигнала: {signal.min()} до {signal.max()}")
        print(f"Длительность сигнала: {signal.shape[1]/sample_rate}с")
        return signal, sample_rate


class WhiteNoise(AudioStream):
    """
    Класс для работы с белым шумом.
    """
    def __init__(self, duration, sample_rate=44100, amplitude=0.1):
        self.duration = duration
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        name = f"white_noise_{duration}s_{sample_rate}hz"
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
    
    # Загрузка и анализ аудиофайла
    audio = Audio("data/00001.wav")
    signal, sample_rate = audio.load()
    audio.analyze(signal[0], sample_rate)
    
    # Применение фильтров к аудиофайлу
    filtered_signals = audio.apply_filters(signal[0], sample_rate)
    
    # Загрузка и анализ белого шума
    noise = WhiteNoise(1, sample_rate)
    noise_signal, noise_sr = noise.load()
    noise.analyze(noise_signal, noise_sr)
    
    # Применение фильтров к белому шуму
    filtered_noise = noise.apply_filters(noise_signal, noise_sr)
    
    # Загрузка и анализ другого аудиофайла
    audio2 = Audio("data/аннигиляторная пушка.wav")
    signal2, sample_rate2 = audio2.load()
    audio2.analyze(signal2[0], sample_rate2)

    # Применение фильтров к другому аудиофайлу
    filtered_signals2 = audio2.apply_filters(signal2[0], sample_rate2)

    audio3 = Audio("data/organ.mp3")
    signal3, sample_rate3 = audio3.load()

    
    
    
    plt.ioff()
    plt.show(block=True)

