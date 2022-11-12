import numpy as np  # Manejo de arrays.
import matplotlib.pyplot as plt  # Visualizacion de graficos.
from scipy import signal, stats
import scipy.io as sio
import warnings
import soundfile as sf
plt.style.use(['dark_background'])  # Para gráficas para temas oscuros.
warnings.filterwarnings("ignore")

# Se diseñara un filtro FIR pasabanda usando mínimos cuadrados
# Definimos los parámetros del filtro

fs = 1024  # Hz
nyquist = fs/2
frangos = [20, 45]  # Rangos de corte
valores_transicion = [0.01, 0.053333, 0.096667, 0.14,
                      0.183333333, 0.2266667, 0.27, 0.3133333, 0.35666667, 0.4]
transicion = valores_transicion[0]

# orden= int(9*fs/frangos[0])#modif el mult para reducir la distorcion en el lado izquierdo del filtro  en frecuencia
valores_orden = [50, 128, 206, 283, 361, 439, 517, 594, 572, 750]
orden = valores_orden[9]

# Cambiamos el orden del filtro FIR a uno impar
if orden % 2 == 0:
    orden += 1

# Definimos la plantilla y las frecuencias del filtro teniendo en cuenta la función firls

plantilla = [0, 0, 1, 1, 0, 0]
frecuencias = [0, frangos[0]-frangos[0]*transicion, frangos[0],
               frangos[1], frangos[1]+frangos[1]*transicion, nyquist]

# Calculo del Kernel
kernel = signal.firls(orden, frecuencias, plantilla, fs=fs)

# GENERAMOS LA SEÑAL
signal, samplerate = sf.read('prueba.wav')

# Amplificamos la señal 
n = len(signal)
for i in range(0, n):
    signal[i] = signal[i]*100


signalaudio2 = []
for i in range(n):
    signalaudio2.append(signal[i][0])  # convertir en un canal de audio

# ploteamos la señal
plt.figure(figsize=(15, 6))
plt.plot(signalaudio2, 'r')
plt.xlabel("Muestras")
plt.title('Señal original')
plt.grid()
plt.show()

# ploteamos el kernel

plt.figure(figsize=(15, 6))
plt.plot(kernel, 'r')
plt.xlabel("Muestras")
plt.title(f'KERNEL (FIRLS) - Orden {orden}')
plt.grid()
plt.show()

# Calculamos la respuesta en frecuencia del filtro

espectro_kernel = np.abs(np.fft.fft(kernel))**2

# calculamos el vector de frecuencias del filtro y eliminamos la parte negativa
hz = np.linspace(0, nyquist, int(np.floor(len(kernel)/2)+1))
espectro_kernel = espectro_kernel[0:len(hz)]


# Filtro aplicado a la señal original
a = [1]

signal_filtred = signal.filtfilt(kernel, a, signalaudio2)

sf.write('nuevo_cuadrado.wav', signal_filtred*100, samplerate)


# graficamos el espectro del kernel

plt.figure(figsize=(15, 6))
plt.plot(hz, espectro_kernel, 'bs-', label='Actual')
plt.plot(frecuencias, plantilla, 'ro-', label='Ideal')
plt.xlim([0, frangos[0]*4])
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Ganancia")
plt.legend()
plt.title("Respuesta en frecuencia del filtro (FIRLS)")
plt.grid()
plt.show()

# graficamos el espectro del kernel en escala logarítmica

plt.figure(figsize=(15, 6))
plt.plot(hz, 10*np.log10(espectro_kernel), 'bs-', label='Actual')
plt.plot([frangos[0], frangos[0]], [-40, 5], 'ro-', label='Ideal')
plt.xlim([0, frangos[0]*4])
plt.ylim([-40, 5])
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Ganancia")
plt.legend()
plt.title("Respuesta en frecuencia del filtro - Escala logaritmica (FIRLS)")
plt.grid()
plt.show()


# Señal original con el filtro kernel aplicado

plt.figure(figsize=(15, 6))
plt.plot(signal_filtred, 'r')
plt.xlabel("Muestras")
plt.title('Filtro aplicado a la señal original')
plt.grid()
plt.show()

# Señal original y señal con filtro
plt.figure(figsize=(15, 6))
plt.plot(signalaudio2, 'r-', label='señal original')
plt.plot(signal_filtred, 'b-', linewidth=2, label='señal con filtro')
plt.title(f'Filtro aplicado y señal original')
plt.legend(loc='best')
plt.show()
