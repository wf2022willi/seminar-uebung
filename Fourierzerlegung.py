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

# # Die Fourierzerlegung

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integr


# ### Hie definieren wir uns Sinus- und Kosinusfunktionen, 
# ### die in das Intervall [-L,L] reinpassen

#Definition von Funktionen
def cos(x,n=1,L=1):
    return np.cos(n*np.pi/L*x)
def sin(x,n,L):
    return np.sin(n*np.pi/L*x)


def f(x):
    return x**2-x**4
def F(x):
    return (x**3)/3. - (x**5)/5.


# %matplotlib inline
x = np.linspace(-1,1.,201)
fig, ax = plt.subplots() # let us plot the data
ax.set_xlim([-1, 1])
ax.set_ylim([0, .4])
ax.plot(x,f(x))
#ax.plot(x,y)

x = np.linspace(-1,1,201)

# %matplotlib notebook
fig, ax = plt.subplots(2,2)
ax[0,0].plot(x,x,label='y=x')
ax[0,0].set_ylabel('y')
ax[0,0].legend()
ax[0,0].grid()
ax[0,1].plot(x,x**2)
ax[1,0].plot(x,x**3)
ax[1,1].plot(x,x**4-x**2)

# %matplotlib notebook
fig, ax = plt.subplots()
ax.plot(x,x,label='y=x')
ax.set_ylabel('y')
ax.grid(ls='--')
ax.plot(x,x**2,label='y=x**2')
ax.plot(x,x**3,label='y=x**3')
ax.plot(x,x**4-x**2,label='y=x**4-x**2')
ax.legend()

# # Integration
# Integration mit numpy.trapz

L = 1.
n = 1
m = 1
x = np.linspace(-1.,1.,201)
#plt.plot(x,cos(x,n,L))
#plt.plot(x,cos(x,m,L))
plt.plot(x,cos(x,n,L)*cos(x,m,L))
#np.trapz(cos(x,n,L))

L=1
c = [np.trapz(cos(x,i,L)*f(x),dx=0.01) for i in np.arange(101)]
s = [np.trapz(sin(x,i,L)*f(x),dx=0.01) for i in np.arange(11)]
y_cos = np.array([c[i]*cos(x,i,L) for i in np.arange(101)])
y_cos[0] = y_cos[0]/2

# %matplotlib notebook
fig, (ax1,ax2,ax3) = plt.subplots(3,1)
#
#for i in np.arange(11):
#    ax.plot(y_cos[i])
ax1.plot(x,f(x))
ax2.plot(x,np.sum(y_cos,axis=0))
ax3.plot(c)


np.fft.fft(f(x)).real.shape

s

# # Fourierreihe in 2D
# # A propos Markdown Cell
# # Dies ist eine markdown cell
# ## und das ist kleiner
# ### noch kleiner
# #### noch viel kleiner
# ##### noch viel viel kleiner
# Ab jetzt ohne hashtag
#
# **Fett**
#
# *italic*
#
# 1. a
# 1. b
#
# * a
# * b
#
# Dies ist eine inline Formel $e^{i\pi}+1=0$
#
# Und hier kommt eine agesetzte Formel
#
# \begin{equation}
# i\hbar\frac{\partial\Psi}{\partial t}=\hat{H}\Psi
# \end{equation}

x = np.linspace(-1.,1.,201)
y = np.linspace(-1.,1.,201)
X,Y = np.meshgrid(x,y)
A = np.exp(-(X**2+Y**2))

# %matplotlib notebook
#
plt.imshow(A)

# %matplotlib notebook
FTA = np.fft.fft2(A)
FTAinv = np.fft.ifft2(A)
plt.imshow(FTAinv.real)




