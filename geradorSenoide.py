import numpy as np
import scipy.io.wavfile as wav

# gerador de senoides de qualquer frequencia com amplitude 1 e taxa de amostragem de 2kHz

def gerar_senoide(frequencia, taxa_amostragem, duração, nome_arquivo):
    # Tempo
    t = np.linspace(0, duração, int(taxa_amostragem * duração), endpoint=False)
    
    # Gere a senoide
    sinal = np.sin(2 * np.pi * frequencia * t)
    
    # Normaliza o sinal
    sinal = sinal / np.max(np.abs(sinal))
    
    # Converter os valores para inteiros de 16 bits
    sinal_int = np.int16(sinal * 32767)
    
    # Salvar o sinal em um arquivo WAV
    wav.write(nome_arquivo, taxa_amostragem, sinal_int)
    
    print(f"Arquivo {nome_arquivo} gerado com sucesso.")

# Parâmetros comuns
taxa_amostragem = 2000
duração = 1.0

frequencia_1 = 1000
nome_arquivo_1 = "senoide_1kHz.wav"
gerar_senoide(frequencia_1, taxa_amostragem, duração, nome_arquivo_1)

frequencia_2 = 700
nome_arquivo_2 = "senoide_235Hz.wav"
gerar_senoide(frequencia_2, taxa_amostragem, duração, nome_arquivo_2)
