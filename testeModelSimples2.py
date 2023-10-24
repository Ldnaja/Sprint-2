import numpy as np
import scipy.signal as signal
import soundfile as sf

# Função para analisar um arquivo de áudio com uma senoide
def analyze_audio_file(audio_file):
    # Carregue o arquivo de áudio
    signal_data, fs = sf.read(audio_file)

    # Calcule a Transformada Rápida de Fourier (FFT) para obter a frequência dominante
    fft_result = np.fft.fft(signal_data)
    freqs = np.fft.fftfreq(len(fft_result), 1 / fs)
    dominant_frequency = freqs[np.argmax(np.abs(fft_result))]

    # Calcule a amplitude máxima
    amplitude = np.max(np.abs(signal_data))

    # Calcule a fase
    phase = np.angle(signal_data[0])

    return dominant_frequency, amplitude, phase

# Caminho para o arquivo de áudio a ser analisado
audio_file_path = 'audio_samples/sine_wave_912.wav'

# Chame a função para analisar o arquivo de áudio
dominant_frequency, amplitude, phase = analyze_audio_file(audio_file_path)

print(f"Frequência Dominante: {dominant_frequency} Hz")
print(f"Amplitude: {amplitude}")
print(f"Fase: {phase}")