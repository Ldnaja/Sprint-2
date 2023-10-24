import tensorflow as tf
import numpy as np
import librosa

# teste de frequencia com o modelo melhorado

# Carregue o modelo treinado
model = tf.keras.models.load_model('modelMelhorado')  # Substitua pelo nome do arquivo de modelo

# Defina a taxa de amostragem apropriada para o seu arquivo de áudio
sample_rate = 2000

# Carregue o arquivo de áudio a ser classificado
audio_file = 'senoide_235Hz.wav'  # Substitua pelo nome do seu arquivo de áudio
audio, _ = librosa.load(audio_file, sr=sample_rate)

# Certifique-se de que o arquivo de áudio tenha a mesma duração (1 segundo) que os dados de treinamento
if len(audio) != sample_rate:
    raise ValueError("O arquivo de áudio deve ter a mesma duração que os dados de treinamento (1 segundo).")

# Expanda as dimensões do áudio para corresponder ao formato de entrada da LSTM
audio = np.expand_dims(audio, axis=0)
audio = np.expand_dims(audio, axis=2)

# Faça a previsão da frequência
predicted_frequency = model.predict(audio)

# Desnormalize a previsão (se você normalizou os rótulos durante o treinamento)
predicted_frequency = predicted_frequency * 500 + 500

print(f"A frequência prevista é {predicted_frequency[0][0]:.2f} Hz")
