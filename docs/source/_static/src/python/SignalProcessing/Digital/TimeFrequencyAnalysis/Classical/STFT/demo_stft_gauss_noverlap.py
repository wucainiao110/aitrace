import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

T = 5
Fs = 400.0
n = T * Fs

f1 = 10
f2 = 25
f3 = 50
f4 = 100

t1 = np.linspace(0, T, n)
x1 = np.cos(2 * np.pi * f1 * t1)
t2 = np.linspace(T, 2 * T, n)
x2 = np.cos(2 * np.pi * f2 * t2)
t3 = np.linspace(2 * T, 3 * T, n)
x3 = np.cos(2 * np.pi * f3 * t3)
t4 = np.linspace(3 * T, 4 * T, n)
x4 = np.cos(2 * np.pi * f4 * t4)

x = np.hstack((x1, x2, x3, x4))
t = np.hstack((t1, t2, t3, t4))

TW = 1000e-3
NW = int(TW * Fs)
print(NW)

nperseg = NW


nfft = 2048

window = ('gaussian', 30)


Toverlaps = [10e-3, 40e-3, 80e-3, 100e-3, 400e-3, 800e-3]
noverlaps = []
for To in Toverlaps:
    noverlaps.append(int(To * Fs))
print(noverlaps)


print(t.shape, x.shape)

plt.figure()
plt.plot(t, x, '-r')
plt.grid()
plt.xlabel('Time/s')
plt.ylabel('Amplitude')
plt.title('Time domain signal')
plt.figure()

cnt = 1
for noverlap in noverlaps:

    # (x, fs=1.0, window=('tukey', 0.25), nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd')
    # [F, T, S] = signal.spectral.spectrogram(x, np.hamming(1024), nperseg=1024, noverlap=0, detrend=False, return_onesided=True, mode='magnitude')
    [F, T, S] = signal.spectral.spectrogram(x, fs=Fs, window=window, nperseg=nperseg, noverlap=noverlap,
                                            nfft=nfft, detrend=False, return_onesided=True,
                                            scaling='density', mode='psd')

    print('T: ', T.shape, 'F:', F.shape, 'S:', S.shape)
    print(S.shape)

    plt.subplot(2, 3, cnt)
    map = plt.pcolormesh(T, F, S, cmap='jet')
    plt.ylabel('Frequency/Hz')
    plt.xlabel('Time/s')
    plt.axis('tight')
    plt.title('STFT->' + 'noverlap:' + str(Toverlaps[cnt-1]) + 's')
    cb = plt.colorbar(cax=None, ax=None, shrink=0.5)
    cnt = cnt + 1
plt.show()
