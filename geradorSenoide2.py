import numpy as np
import scipy.signal as signal
import soundfile as sf

# Parâmetros do sinal
fs = 2000  # Frequência de amostragem
duration = 1  # Duração em segundos
num_samples = fs * duration

# Número de exemplos de treinamento
num_samples = 1000

# Diretório para salvar os arquivos de áudio
output_directory = 'audio_samples/'

# Função para gerar senoides com variações de amplitude e fase
def generate_sine_wave(frequency, amplitude_variation, phase_variation):
    t = np.linspace(0, duration, num_samples, endpoint=False)
    amplitude = 1.0 + amplitude_variation * np.random.randn()
    phase = phase_variation * np.random.randn()
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return sine_wave

for i in range(num_samples):
    if np.random.rand() < 0.5:
        frequency = 500
    else:
        frequency = 1000

    amplitude_variation = 0.2 * np.random.randn()  # Variação da amplitude
    phase_variation = 0.2 * np.random.randn()  # Variação da fase

    sine_wave = generate_sine_wave(frequency, amplitude_variation, phase_variation)

    # Salve o sinal de áudio em um arquivo WAV
    filename = f'{output_directory}sine_wave_{i}.wav'
    sf.write(filename, sine_wave, fs)

print(f"{num_samples} arquivos de áudio gerados e salvos em '{output_directory}'.")
