# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt

n = 100
T = np.linspace(0.,2.*np.pi,n)
#omega = 2.0*np.pi*np.arange(n//10)
#ampsin = np.linspace(0,5,n//10)
#ampcos = np.linspace(0,5,n//10)
#ampsin[3:] = 0
#ampcos[3:] = 0
y = 1.*T/2./np.pi

# +
# %matplotlib notebook

#for i in np.arange(100):
#    y = y + ampcos[i]*np.cos(omega[i]*T) + ampsin[i]*np.sin(omega[i]*T)  # y+= np.sin(omega[i]*T)
#
plt.plot(T,y)
# -

# %matplotlib notebook
#y=np.random.rand(1000)
f = np.fft.fft(y)
f_shift = np.fft.fftshift(f)
freq = np.fft.fftshift(np.fft.fftfreq(n,1/n))
#
#plt.plot(freq,f.real)
#plt.plot(freq,f.imag)
plt.plot(freq,f_shift.real)
plt.plot(freq,f_shift.imag)

# %matplotlib notebook
g=np.copy(f)
h=np.copy(f)
g[1:10] = 0.+0.j
h[90:] = 0.+0.j
plt.plot(np.fft.ifft(f))
plt.plot(np.fft.ifft(g))
plt.plot(np.fft.ifft(h))
