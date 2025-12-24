import numpy as np
import matplotlib.pyplot as plt

def dft(f):
    N = len(f)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, f)

def idft(F):
    N = len(F)
    k = np.arange(N)
    n = k.reshape((N, 1))
    e = np.exp(2j * np.pi * k * n / N)
    return np.dot(e, F) / N

N = 100
t = np.linspace(0, 1, N, endpoint=False)
f_x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 15 * t)

F_omega = dft(f_x)
f_reconstructed = idft(F_omega)

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t, f_x, label='f(x)')
plt.title('1. Original Signal (Time Domain)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
freqs = np.linspace(0, N, N)
plt.stem(freqs[:N//2], np.abs(F_omega)[:N//2])
plt.title('2. Frequency Spectrum |F(w)| (Frequency Domain)')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, f_reconstructed.real, color='orange', label='Reconstructed')
plt.title('3. Reconstructed Signal (IDFT)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('fourier_transform_plot.png')
