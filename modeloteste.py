# import tensorflow as tf
# import numpy as np

# # Gere dados de treinamento para várias frequências entre 500Hz e 1kHz
# fs = 2000  # Frequência de amostragem
# t = np.linspace(0, 1, fs, endpoint=False)
# num_samples = 100  # Número de amostras por frequência
# frequencies = np.linspace(500, 1000, num_samples)  # Frequências variando de 500Hz a 1kHz

# x_data = []
# y_data = []

# for freq in frequencies:
#     signal = np.sin(2 * np.pi * freq * t)
#     x_data.append(signal)
#     y_data.append(freq)

# # Converta os dados em arrays numpy
# x_data = np.array(x_data)
# y_data = np.array(y_data)

# # Normalize os rótulos para o intervalo [0, 1]
# y_data = (y_data - 500) / 500  # Normaliza para o intervalo [0, 1]

# # Crie um modelo LSTM genérico
# model = tf.keras.Sequential([
#     tf.keras.layers.LSTM(64, input_shape=(fs, 1)),
#     tf.keras.layers.Dense(1, activation='linear')  # Uma saída para a frequência
# ])

# # Compile o modelo
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# # Expanda as dimensões dos dados para corresponder ao formato de entrada da LSTM
# x_data = np.expand_dims(x_data, axis=2)

# # Treine o modelo
# model.fit(x_data, y_data, epochs=100)

# # Salve o modelo
# model.save('modelGenerico')

import tensorflow as tf
import numpy as np

# Gere dados de treinamento para várias frequências entre 500Hz e 1kHz
fs = 2000  # Frequência de amostragem
t = np.linspace(0, 1, fs, endpoint=False)
num_samples = 100  # Número de amostras por frequência
frequencies = np.linspace(500, 1000, num_samples)  # Frequências variando de 500Hz a 1kHz

x_data = []
y_data = []

for freq in frequencies:
    signal = np.sin(2 * np.pi * freq * t)
    x_data.append(signal)
    y_data.append(freq)

# Converta os dados em arrays numpy
x_data = np.array(x_data)
y_data = np.array(y_data)

# Normalize os rótulos para o intervalo [0, 1]
y_data = (y_data - 500) / 500  # Normaliza para o intervalo [0, 1]

# Crie um modelo LSTM mais complexo com regularização
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(fs, 1), return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.01))  # Uma saída para a frequência
])

# Compile o modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

x_data = np.expand_dims(x_data, axis=2)

model.fit(x_data, y_data, epochs=200)

model.save('modelMelhorado') # Modelo com frequencias de 500 a 1000Hz
