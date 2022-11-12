################# IMPORTACIÓN DE LIBRERÍAS ###############
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats 
import scipy.io as sio
import warnings
import soundfile as sf

# Tema oscuro en gráficos.
plt.style.use(['dark_background'])  
warnings.filterwarnings("ignore")

################# Parámetros ####################
fs = 1024  
nyquist = fs/2
frangos = [20, 45]

# Coeficientes del filtro
kernelB, kernelA = signal.butter(4, np.array(frangos)/nyquist, btype='bandpass')

# Espectro de potencial del kernelB
espectro_kernelB = np.abs(np.fft.fft(kernelB))**2

# Vectores de frecuencia
hz = np.linspace(0, nyquist, int(np.floor(len(kernelB)/2)+1))

# Respuesta al impulso y frecuencia de un filtro IIR
signal, samplerate = sf.read('prueba.wav')

n = len(signal)
for i in range(0, n):
    signal[i] = signal[i]/20


signalaudio2 = []
for i in range(n):
    signalaudio2.append(signal[i][0])  # convertir en un canal de audio

# aplicar filtro
fimp = signal.filtfilt(kernelB, kernelA, signalaudio2)
sf.write('nuevo_iir.wav', fimp*100, samplerate)

# Cálculo de espectro
fimpX = np.abs(np.fft.fft(fimp))**2
hz_impulso = np.linspace(0, nyquist, int(np.floor(len(signal)/2)+1))

# Graficando respuesta en frecuencia
plt.figure(figsize=(15, 6))
plt.plot(hz_impulso, fimpX[0:len(hz_impulso)], 'bs-', label='Actual')
plt.plot([0, frangos[0], frangos[0], frangos[1], frangos[1],
         nyquist], [0, 0, 1, 1, 0, 0], 'r', label='Ideal')
plt.xlim([0, 100])
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Atenuación")
plt.legend()
plt.title("Respuesta en frecuencia del filtro IIR")
plt.grid()
plt.show()

# Graficando respuesta en frecuencia en escala logarítmica
plt.figure(figsize=(15, 6))
plt.plot(hz_impulso, 10*np.log10(fimpX[0:len(hz_impulso)]), 'bs-')
plt.xlim([0, 100])
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Atenuación")
plt.legend()
plt.title("Respuesta en frecuencia del filtro IIR - Escala logaritmica")
plt.grid()
plt.show()

# Graficando
plt.figure(figsize=(15, 6))
plt.plot(signalaudio2, 'b', label='Señal original')
plt.plot(fimp, 'r', label='Filtro de la señal')
plt.xlim([1, len(signal)])
plt.xlabel("Muestras")
plt.legend()
plt.title(f'Filtro aplicado y señal original')
plt.grid()
plt.show()