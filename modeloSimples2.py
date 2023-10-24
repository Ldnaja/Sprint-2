import tensorflow as tf
import numpy as np

fs = 2000  # Frequência de amostragem

# Função para gerar senoides com variações de amplitude e fase
def generate_sine_wave(frequency, amplitude_variation, phase_variation, fs = fs):
    t = np.linspace(0, 1, fs, endpoint=False)
    amplitude = 1.0 + amplitude_variation * np.random.randn()
    phase = phase_variation * np.random.randn()
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return sine_wave

# Número de exemplos de treinamento
num_samples = 500

# Gere dados de treinamento (senoides de 500Hz e 1kHz) com variações de amplitude e fase
x_train = []
y_train = []

for _ in range(num_samples):
    if np.random.rand() < 0.5:
        frequency = 500
    else:
        frequency = 1000

    amplitude_variation = 0.2 * np.random.randn()  # Variação da amplitude
    phase_variation = 0.2 * np.random.randn()  # Variação da fase

    sine_wave = generate_sine_wave(frequency, amplitude_variation, phase_variation)
    x_train.append(sine_wave)
    if frequency == 500:
        y_train.append([1, 0])
    else:
        y_train.append([0, 1])

x_train = np.array(x_train)
y_train = np.array(y_train)

# Crie um modelo LSTM
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(fs, 1)),
    tf.keras.layers.Dense(2, activation='softmax')  # Duas classes: 500Hz e 1kHz
])

# Compile o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Expanda as dimensões dos dados para corresponder ao formato de entrada da LSTM
x_train = np.expand_dims(x_train, axis=2)

# Treine o modelo
model.fit(x_train, y_train, epochs=50)

# Salve o modelo
model.save('model_with_amplitude_and_phase_variation')
