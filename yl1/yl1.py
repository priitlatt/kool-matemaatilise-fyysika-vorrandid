
# coding: utf-8

# # MTMM.00.209 Matemaatilise füüsika võrrandid #
# 
# ## 1. kodutöö - Priit Lätt ##

# Impordime edaspidises vajaminevad moodulid `numpy` ja `matplotlib`.

# In[1]:

import numpy as np
import matplotlib.pyplot as plt


# Defineerime konstandid ja muutujad.

# In[2]:

K = 0.25  # diffusioonikordaja
T = 5 # aeg
n = 100  # punktide arv

dely = 2.0 / n
delx = 0.01
x = np.arange(-10, 10, delx)
y = np.arange(-1 + 0.5*dely, 1 - 0.5*dely, dely)


# In[3]:

get_ipython().run_cell_magic(u'latex', u'', u'Defineerime algfunktsiooni $f(x) = \\frac{\\sin x}{x}$.')


# In[4]:

f = lambda x: np.sin(x)/x


# Failist [`heat1.m`](http://www2.math.umd.edu/~jcooper/PDEbook/chap3/heat1.m) meie jaoks oluline osa, mis arvutab funktsioonile vastava difusioonivõrrandi lahendi.

# In[5]:

def diffusioon(t, k, algfun=f):
    s = np.zeros(len(x))
    for _y in y:
        s = s + np.exp( -(x - _y)**2 / (4*k*t) ) * algfun(_y)
    snap1 = (1.0 / np.sqrt(4*np.pi*k*t)) * s * dely
    return snap1


# Leiame difusioonivõrrandi lahendid fikseeritud difusioonikorjajaga.

# In[6]:

snaps_1 = [(diffusioon(i, K), i, K) for i in range(1, 9)]


# Joonistame välja difusioonivõrrandite lahendite graafikud fikseeritud difusioonikordaja jaoks.

# In[7]:

plt.figure(figsize=(20, 10))
plt.rcParams['font.size'] = 18

plt.title("Difusioonivorrandi lahendid fikseeritud difusioonikordajaga")

plt.plot(x, map(f, x), '-.', label=r"algfuntsioon", linewidth=2.0)
for snap, i, k in snaps_1:
    label = "$t = %.1f,\ k = %.2f$" % (i, k)
    plt.plot(x, snap, label=label)

plt.legend()

plt.show()


# Leiame difusioonivõrrandi lahendid fikseeritud aja korral.

# In[8]:

snaps_2 = [(diffusioon(T, i / 5.0), T, i/5.0) for i in range(1, 9)]


# Joonistame välja difusioonivõrrandite lahendite graafikud fikseeritud aja jaoks.

# In[9]:

plt.figure(figsize=(20, 10))
plt.rcParams['font.size'] = 18

plt.title("Difusioonivorrandi lahendid fikseeritud aja korral")

plt.plot(x, map(f, x), '-.', label=r"algfuntsioon", linewidth=2.0)
for snap, i, k in snaps_2:
    label = "$t = %.1f,\ k = %.1f$" % (i, k)
    plt.plot(x, snap, label=label)

plt.legend()

plt.show()


# Leiame difusioonivõrrandi lahendid erinevate algfunktsioonide korral.

# In[10]:

algfunktsioonid = [
    (lambda x: -3*f(x), "-3f(x)"),
    (lambda x: -2*f(x), "-2f(x)"),
    (lambda x: -1*f(x), "-f(x)"),
    (lambda x: -0.5*f(x), r"-\frac{f(x)}{2}"),
    (lambda x: 0.5*f(x), r"\frac{f(x)}{2}"),
    (lambda x: f(x), "f(x)"),
    (lambda x: 2*f(x), "2f(x)"),
    (lambda x: 3*f(x), "3f(x)"),
]

snaps_3 = [(diffusioon(T, K, algfun=fun), label) for fun, label in algfunktsioonid]


# Joonistame välja difusioonivõrrandite lahendite graafikud erinevate algfunktsioonide jaoks.

# In[11]:

plt.figure(figsize=(20, 10))
plt.rcParams['font.size'] = 18

plt.title("Difusioonivorrandi lahendid erinevate algfunktsioonide korral")

for snap, label in snaps_3:
    plt.plot(x, snap, label="${}$".format(label))

plt.legend()

plt.show()

