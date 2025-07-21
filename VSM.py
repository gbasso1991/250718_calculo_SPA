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
filename='NE@250609_C.txt'
data = np.loadtxt(filename, skiprows=12)

H = data[:, 0]  # Gauss
m = data[:, 1]  # emu

fig1, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
ax.plot(H, m, '.-', label=filename)

for a in [ax]:
    a.legend(ncol=1)
    a.grid()
    a.set_ylabel('m (emu)')
plt.xlabel('H (G)')
plt.show()
#%% 
C_conc = 9.7 #g/L
C_conc_mm = C_conc/1000 # uso densidad del H2O 1000 g/L

masa_FF=(0.1068-0.0570) #g
m_norm =(m/masa_FF)/C_conc_mm  # emu/g

fig1, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
#ax.plot(H_pat, m_pat, '-', label='Patron')
ax.plot(H, m_norm, '-', label='NE')

for a in [ax]:
    a.legend(ncol=1)
    a.grid()
    a.set_ylabel('m (emu/g)')
plt.xlabel('H (G)')
plt.show()

# %% Anhisteretico para fit
H_norm_ah,m_norm_ah = mt.anhysteretic(H, m_norm)

fit = fit3.session(H_norm_ah, m_norm_ah, fname='NE@citrato_250609_C', divbymass=False)
fit.fix('sig0')
fit.fix('mu0')
fit.free('dc')
fit.fit()
fit.update()
fit.free('sig0')
fit.free('mu0')
fit.set_yE_as('sep')
fit.fit()
fit.update()
fit.save()
fit.print_pars()
H_fit = fit.X
m_fit = fit.Y
m_fit_sin_diamag = m_fit - lineal(H_fit, fit.params['C'].value, fit.params['dc'].value)

# %%

fig1, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
#ax.plot(H_pat, m_pat, '-', label='Patron')
ax.plot(H, m_norm, '-', label=filename[:-4])
ax.plot(H_fit, m_fit, '.-', label='fit')
ax.plot(H_fit, m_fit_sin_diamag, '-', label='fit dm')
for a in [ax]:
    a.legend(ncol=1)
    a.grid()
    a.set_ylabel('m (emu/g)')
plt.xlabel('H (G)')
plt.savefig('ciclo_vsm.png',dpi=300)
plt.show()
# %%
Ms=ufloat(np.mean([max(m_fit_sin_diamag),-min(m_fit_sin_diamag)]),np.std([max(m_fit_sin_diamag),-min(m_fit_sin_diamag)]))*1000

print(f'Ms = {Ms} A/m')

# %% 1. Cálculo de constantes físicas y valor de x

import numpy as np

mu0 = np.pi * 4e-7          # Permeabilidad del vacío
kB  = 1.38e-23              # Constante de Boltzmann
T   = 300                   # Temperatura en K
mB  = 9.27e-24              # Magnetón de Bohr
mu  = 166356 * mB           # Momento magnético total
H0  = 57e3                  # Campo magnético externo

# Cálculo del parámetro adimensional x
x = mu0 * mu * H0 / (kB * T)
print(f'x = {x:.2e}')

# %% 2. Sumatoria simbólica con SymPy

from sympy import symbols, Sum, oo, sqrt, pi

x_sym, k = symbols('x k', real=True, positive=True)

# Expresión dentro de la sumatoria
expr = 1 - 1 / sqrt(1 + (x_sym**2) / (pi * k**2))

# Sumatoria infinita
suma_simbolica = Sum(expr, (k, 1, oo))

# Expresión completa
formula = (4 / x_sym) * suma_simbolica

# Mostrar fórmula simbólica
print("Expresión simbólica:")
display(formula)
# %% 3. Evaluación numérica truncada para N = 1000

from math import sqrt, pi

N = 1000

suma = sum(1 - 1 / sqrt(1 + (x**2) / (pi * k**2)) for k in range(1, N + 1))
alpha1 = (4 / x) * suma

print(f"Resultado con N = {N} términos: alpha1 ≈ {alpha1:.10f}")

# %% 4. Cálculo de α₁ para valores crecientes de N y gráfico
import matplotlib.pyplot as plt

# Valores crecientes de N
N_values = [10**3,2*10**3,4*10**3, 5*10**3, 10**4,5*10**4, 10**5, 10**6,5*10**6,10**7]
alpha_values = []

for N in N_values:
    suma = sum(1 - 1 / sqrt(1 + (x**2) / (pi * k**2)) for k in range(1, N + 1))
    alpha = (4 / x) * suma
    alpha_values.append(alpha)
    print(f"N = {N:<7} → alpha1 ≈ {alpha:.10f}")

# %%  5. Gráfico de convergencia de α₁(N)

plt.figure(figsize=(8, 5), constrained_layout=True)
ax = plt.gca()

# Gráfico de puntos
ax.plot(N_values, alpha_values, 'o-')
ax.set_xscale('log')
ax.set_xlabel("Número de términos (N)")
ax.set_ylabel(r"$\alpha_1$")
ax.set_title(r"Convergencia de $\alpha_1$ con $N$")
ax.grid(True)

# Último valor calculado
last_N = N_values[-1]
last_alpha = alpha_values[-1]

# Texto con valor numérico final
ax.text(
    0.6, 0.5,
    rf"$\alpha_1 \approx {last_alpha:.6f}$" + f"(N = {last_N:.0e})",
    transform=ax.transAxes,
    fontsize=14, ha='center', va='center',
    bbox=dict(facecolor='tab:blue', alpha=0.8, boxstyle='round')
)

# Texto con expresión simbólica de alpha1
ax.text(
    0.6, 0.3,
    r"$\alpha_1(x) = \frac{4}{x} \sum_{k=1}^\infty \left(1 - \frac{1}{\sqrt{1 + \frac{x^2}{\pi k^2}}}\right)$",
    transform=ax.transAxes,
    fontsize=13,
    ha='center', va='center',
    bbox=dict(facecolor='white', alpha=0.85, boxstyle='round')
)
plt.savefig('alfa1.png',dpi=300)
plt.show()
#%% 6. Calculo SPA a 135 kHz 
w = 2*pi*np.mean([135565,135600,135590])          # frecuencia angular rad/s
C = 9.7                 # concentracion kg/m³==g/L
tau= np.mean([54.6,47.8,76.5])*1e-9

primer_factor= 2*Ms*kB*T*w/C/mu
factor_resonante= (w*tau)/(1+(w*tau)**2)
SPA= primer_factor*factor_resonante*last_alpha/1000
print(f'SPA calculado = {SPA:.1f} W/g')

# %%
SAR=np.mean([87,78,121]) #W/g
print(f'SAR de medidas = {SAR:.1f} W/g')

# %%
