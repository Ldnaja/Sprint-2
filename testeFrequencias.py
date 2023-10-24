import tensorflow as tf
import numpy as np
import librosa

# teste com o modelo mais simples

model = tf.keras.models.load_model('modelSimples')

def classify_frequency(audio_file):
    audio, sr = librosa.load(audio_file, sr=2000)  # Certifique-se de que a taxa de amostragem (sr) seja a mesma que usada durante o treinamento

    # Expanda as dimensões do sinal de áudio
    audio = np.expand_dims(audio, axis=0)
    audio = np.expand_dims(audio, axis=2)

    # Faça a previsão com o modelo
    predictions = model.predict(audio)

    threshold = 0.5
    if predictions[0, 0] > threshold:
        return "Frequência detectada: 500Hz"
    elif predictions[0, 1] > threshold:
        return "Frequência detectada: 1kHz"
    else:
        return "Frequência não identificada"

audio_file = 'senoide_1kHz.wav'
result = classify_frequency(audio_file)
print(result)
