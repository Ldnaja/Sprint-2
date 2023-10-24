import tensorflow as tf
import numpy as np

# Gere dados de treinamento (senoides de 500Hz e 1kHz)
fs = 2000  # Frequência de amostragem
t = np.linspace(0, 1, fs, endpoint=False)
x_500Hz = np.sin(2 * np.pi * 500 * t)
x_1kHz = np.sin(2 * np.pi * 1000 * t)

y_500Hz = np.array([1, 0])  # Rótulo para 500Hz
y_1kHz = np.array([0, 1])  # Rótulo para 1kHz

# Crie um modelo LSTM
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(fs, 1)),
    tf.keras.layers.Dense(2, activation='softmax')  # Duas classes: 500Hz e 1kHz
])

# Compile o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Expande as dimensões dos dados de treinamento
x_500Hz = np.expand_dims(x_500Hz, axis=0)
x_500Hz = np.expand_dims(x_500Hz, axis=2)
x_1kHz = np.expand_dims(x_1kHz, axis=0)
x_1kHz = np.expand_dims(x_1kHz, axis=2)

# Combina os dados de treinamento
x_train = np.vstack([x_500Hz, x_1kHz])
y_train = np.vstack([y_500Hz, y_1kHz])

# Treina
model.fit(x_train, y_train, epochs=100)

# Salve o modelo
model.save('modelSimples') # modelo mais simples contendo as frequencias de 500Hz e 1kHz
