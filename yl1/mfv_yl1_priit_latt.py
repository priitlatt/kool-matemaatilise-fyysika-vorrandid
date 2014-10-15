# coding: utf-8

# MTMM.00.209 Matemaatilise füüsika võrrandid
# 1. kodutöö - Priit Lätt

# Impordime edaspidises vajaminevad moodulid `numpy` ja `matplotlib`.
import numpy as np
import matplotlib.pyplot as plt

# Defineerime konstandid ja muutujad.
K = 0.25  # diffusioonikordaja
T = 5  # aeg
n = 100  # punktide arv

dely = 2.0 / n
delx = 0.01
x = np.arange(-5, 5, delx)
y = np.arange(-1 + 0.5*dely, 1 - 0.5*dely, dely)

# Defineerime algfunktsiooni f(x) = sin(x)/x.
f = lambda x: np.sin(x)/x


# Failist `heat1.m` (http://www2.math.umd.edu/~jcooper/PDEbook/chap3/heat1.m)
# meie jaoks oluline osa, mis arvutab funktsioonile vastava difusioonivõrrandi
# lahendi.
def diffusioon(t, k, algfun=f):
    s = np.zeros(len(x))
    for _y in y:
        s = s + np.exp(-(x - _y)**2 / (4*k*t)) * algfun(_y)
    snap1 = (1.0 / np.sqrt(4*np.pi*k*t)) * s * dely
    return snap1


# Leiame difusioonivõrrandi lahendid fikseeritud difusioonikorjajaga.
snaps_1 = [(diffusioon(i, K), i, K) for i in range(1, 9)]
# Leiame difusioonivõrrandi lahendid fikseeritud aja korral.
snaps_2 = [(diffusioon(T, i / 5.0), T, i/5.0) for i in range(1, 9)]
# Leiame difusioonivõrrandi lahendid erinevate algfunktsioonide korral.
funktsioonid = [
    (lambda x: -3*f(x), "-3f(x)"),
    (lambda x: -2*f(x), "-2f(x)"),
    (lambda x: -1*f(x), "-f(x)"),
    (lambda x: -0.5*f(x), r"-\frac{f(x)}{2}"),
    (lambda x: 0.5*f(x), r"\frac{f(x)}{2}"),
    (lambda x: f(x), "f(x)"),
    (lambda x: 2*f(x), "2f(x)"),
    (lambda x: 3*f(x), "3f(x)"),
]
snaps_3 = [(diffusioon(T, K, algfun=_f), label) for _f, label in funktsioonid]

# Initsialiseerime graafiku
plt.figure(1, figsize=(16, 17))
plt.rcParams['font.size'] = 10


# Kanname alamgraafikule difusioonivõrrandite lahendid fikseeritud
# difusioonikordaja jaoks.
plt.subplot(311)
plt.title("Difusioonivorrandi lahendid fikseeritud difusioonikordajaga")

plt.plot(x, map(f, x), '-.', label=r"algfuntsioon", linewidth=2.0)

for snap, i, k in snaps_1:
    label = "$t = %.1f,\ k = %.2f$" % (i, k)
    plt.plot(x, snap, label=label)

plt.legend()


# Kanname alamgraafikule difusioonivõrrandite lahendid fikseeritud aja jaoks.
plt.subplot(312)
plt.title("Difusioonivorrandi lahendid fikseeritud aja korral")

plt.plot(x, map(f, x), '-.', label=r"algfuntsioon", linewidth=2.0)

for snap, i, k in snaps_2:
    label = "$t = %.1f,\ k = %.1f$" % (i, k)
    plt.plot(x, snap, label=label)

plt.legend()


# Kanname alamgraafikule difusioonivõrrandite lahendid erinevate
# algfunktsioonide jaoks.
plt.subplot(313)
plt.title("Difusioonivorrandi lahendid erinevate algfunktsioonide korral")

for snap, label in snaps_3:
    plt.plot(x, snap, label="${}$".format(label))

plt.legend()

# Kuvame graafiku
plt.show()
