import numpy as np  # Manejo de arrays.
import matplotlib.pyplot as plt  # Visualizacion de graficos.
from scipy import signal, stats
import scipy.io as sio
import warnings
import soundfile as sf

plt.style.use(['dark_background'])  # Para gráficas para temas oscuros.
warnings.filterwarnings("ignore")

# Leemos la señal de audio
signal, samplerate = sf.read('prueba.wav')

# Amplificamos la señal por 100
n = len(signal)
for i in range(0, n):
    signal[i] = signal[i] * 100

signalaudio2 = []
for i in range(n):
    signalaudio2.append(signal[i][0])  # convertir en un canal de audio

# Diseñaremos un filtro FIR pasabanda usando metodo de ventaneo

fs = 1024  # Hz
nyquist = fs / 2
frangos = [25, 45]  # Rangos de corte
transicion = 0.01

# orden= int(9*fs/frangos[0])#modif el mult para reducir la distorcion en el lado izquierdo del filtro  en frecuencia
orden = int(15 * fs / frangos[0])

# Cambiamos el orden a impar para el filtro FIR
if orden % 2 == 0:
    orden += 1

# ploteamos la señal
plt.figure(figsize=(15, 6))
plt.plot(signalaudio2, 'r')
plt.xlabel("Muestras")
plt.title('Señal original')
plt.grid()
plt.show()

# Calculo del Kernel
# pass_zero=False para que sea pasa banda
kernel = signal.firwin(orden, frangos, fs=fs, pass_zero="bandpass")

# ploteamo el kernel

plt.figure(figsize=(15, 6))
plt.plot(kernel, 'r')
plt.xlabel("Tiempo")
plt.title(f'KERNEL (FIRWIN) - Orden {orden}')
plt.grid()
plt.show()

# Calculamos la respuesta en frecuencia del filtro

espectro_kernel = np.abs(np.fft.fft(kernel)) ** 2

# Calculamos el vector de frecuencias del filtro y eliminamos la parte negativa
hz = np.linspace(0, nyquist, int(np.floor(len(kernel) / 2) + 1))
espectro_kernel = espectro_kernel[0:len(hz)]

# Filtro aplicado a la señal original
a = [1]

signal_filtred = signal.filtfilt(kernel, a, signalaudio2)

sf.write('nuevo_ventaneo.wav', signal_filtred, samplerate)

# graficamos el espectro del kernel

plt.figure(figsize=(15, 6))
plt.plot(hz, espectro_kernel, 'bs-', label='Actual')
plt.plot([0, frangos[0], frangos[0], frangos[1], frangos[1],
          nyquist], [0, 0, 1, 1, 0, 0], 'ro-', label='Ideal')
plt.xlim([0, frangos[0] * 4])
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Ganancia")
plt.legend()
plt.title("Respuesta en frecuencia del filtro (FIRwin)")
plt.grid()
plt.show()

# Grafico el espectro del kernel en escala logartimica

plt.figure(figsize=(15, 6))
plt.plot(hz, 10 * np.log10(espectro_kernel), 'bs-', label='Actual')
plt.plot([frangos[0], frangos[0]], [-100, 5], 'ro-', label='Ideal')
plt.xlim([0, frangos[0] * 4])
plt.ylim([-80, 5])
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Ganancia")
plt.legend()
plt.title("Respuesta en frecuencia del filtro - Escala logartimica (FIRwin)")
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
plt.plot(signalaudio2, 'r-', label='Señal original')
plt.plot(signal_filtred, 'b-', linewidth=2, label='señal con filtro')
plt.legend(loc='best')
plt.show()