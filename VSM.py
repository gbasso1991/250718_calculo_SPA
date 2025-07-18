#%% VSM NEcitrato 250613_c
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
from sklearn.metrics import r2_score 
from mlognormfit import fit3
from mvshtools import mvshtools as mt
import re
from uncertainties import ufloat
#%%
def lineal(x,m,n):
    return m*x+n

def coercive_field(H, M):
    """
    Devuelve los valores de campo coercitivo (Hc) donde la magnetización M cruza por cero.
    
    Parámetros:
    - H: np.array, campo magnético (en A/m o kA/m)
    - M: np.array, magnetización (en emu/g)
    
    Retorna:
    - hc_values: list de valores Hc (puede haber más de uno si hay múltiples cruces por cero)
    """
    H = np.asarray(H)
    M = np.asarray(M)
    hc_values = []

    for i in range(len(M)-1):
        if M[i]*M[i+1] < 0:  # Cambio de signo indica cruce por cero
            # Interpolación lineal entre (H[i], M[i]) y (H[i+1], M[i+1])
            h1, h2 = H[i], H[i+1]
            m1, m2 = M[i], M[i+1]
            hc = h1 - m1 * (h2 - h1) / (m2 - m1)
            hc_values.append(hc)

    return hc_values
#%% Levanto Archivos
data = np.loadtxt('NE@citrato250613.txt', skiprows=12)

H = data[:, 0]  # Gauss
m = data[:, 1]  # emu

fig1, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
ax.plot(H, m, '.-', label='8A seco')

for a in [ax]:
    a.legend(ncol=1)
    a.grid()
    a.set_ylabel('m (emu)')
plt.xlabel('H (G)')
plt.show()
#%% 
C_conc = 9.7 #g/L
C_conc_mm = C_conc/1000 # uso densidad del H2O 1000 g/L

masa_FF=(0.1191-0.0685) #g
m_norm =(m/masa_FF)/C_conc_mm  # emu/g

fig1, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
#ax.plot(H_pat, m_pat, '-', label='Patron')
ax.plot(H, m_norm, '-', label='8A')

for a in [ax]:
    a.legend(ncol=1)
    a.grid()
    a.set_ylabel('m (emu/g)')
plt.xlabel('H (G)')
plt.show()


# %%
