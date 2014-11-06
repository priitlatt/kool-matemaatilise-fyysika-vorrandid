# coding: utf-8

# MTMM.00.209 Matemaatilise füüsika võrrandid
# 2. kodutöö - Priit Lätt

# Impordime edaspidises vajaminevad moodulid numpy, scipy ja matplotlib.

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy

# Defineerime konstandid ja muutujad.

delt = 0.05
N = np.array([10, 15, 30, 40])
n1, n2, n3, n4 = N
delx = 0.05
x = np.arange(0, 10+delx, delx)
J = int(10 / delx)


# heat3.m konverteering Pythonisse

def get_snaps(k, func, left, right):
    r = 0.5*k*delt / (delx**2)

    snap0 = func(x)
    v = snap0[1:J:1]

    D = np.array([[1+2*r if i == j else 0 for i in range(J-1)]
                 for j in range(J-1)])
    E = np.array([[-r if j == i+1 else 0 for i in range(J-1)]
                 for j in range(J-1)])
    A = D + E + E.T

    P, L, U = scipy.linalg.lu(A)

    for n in range(n1):
        b = 2*v
        b[0] = b[0] + r*(left(n*delt) + left((n-1)*delt))
        b[-1] = b[-1] + r*(right(n*delt) + right((n-1)*delt))
        y = scipy.linalg.solve(L, b.T)
        z = scipy.linalg.solve(U, y).T
        v = z - v

    snap1 = np.concatenate((left(N*delt), v, right(N*delt)))

    for n in range(n1+1, n1+n2+1):
        b = 2*v
        b[0] = b[0] + r*(left(n*delt) + left((n+1)*delt))
        b[-1] = b[-1] + r*(right(n*delt) + right((n+1)*delt))
        y = scipy.linalg.solve(L, b.T)
        z = scipy.linalg.solve(U, y).T
        v = z - v

    snap2 = np.concatenate((left(N*delt), v, right(N*delt)))

    for n in range(n1+n2+1, n1+n2+n3+1):
        b = 2*v
        b[0] = b[0] + r*(left(n*delt) + left((n+1)*delt))
        b[-1] = b[-1] + r*(right(n*delt) + right((n+1)*delt))
        y = scipy.linalg.solve(L, b.T)
        z = scipy.linalg.solve(U, y).T
        v = z - v

    snap3 = np.concatenate((left(N*delt), v, right(N*delt)))

    for n in range(n1+n2+n3+1, n1+n2+n3+n4+1):
        b = 2*v
        b[0] = b[0] + r*(left(n*delt) + left((n+1)*delt))
        b[-1] = b[-1] + r*(right(n*delt) + right((n+1)*delt))
        y = scipy.linalg.solve(L, b.T)
        z = scipy.linalg.solve(U, y).T
        v = z - v

    snap4 = np.concatenate((left(N*delt), v, right(N*delt)))
    # teeme snap0 sama pikaks kui teised
    nones = [None for _ in range(len(N))]
    snap0 = np.concatenate((nones, snap0, nones))

    return (snap0, snap1, snap2, snap3, snap4)


# heat4.m konverteering Pythonisse

def get_snaps2(k, func, q):
    r = 0.5*k*delt / (delx**2)
    snap0 = func(x)
    qq = q(x)

    v = snap0

    D = np.array([[1+2*r if i == j else 0 for i in range(J+1)]
                 for j in range(J+1)])
    D[0][0] = D[0][0] - r
    D[J][J] = D[J][J] - r
    E = np.array([[-r if j == i+1 else 0 for i in range(J+1)]
                  for j in range(J+1)])

    A = D + E + E.T
    P, L, U = scipy.linalg.lu(A)

    for n in range(n1):
        b = 2*v + delt*qq
        y = scipy.linalg.solve(L, b.T)
        z = scipy.linalg.solve(U, y).T
        v = z - v

    snap1 = v

    for n in range(n1+1, n1+n2+1):
        b = 2*v + delt*qq
        y = scipy.linalg.solve(L, b.T)
        z = scipy.linalg.solve(U, y).T
        v = z - v

    snap2 = v

    for n in range(n1+n2+1, n1+n2+n3+1):
        b = 2*v + delt*qq
        y = scipy.linalg.solve(L, b.T)
        z = scipy.linalg.solve(U, y).T
        v = z - v

    snap3 = v

    for n in range(n1+n2+n3+1, n1+n2+n3+n4+1):
        b = 2*v + delt*qq
        y = scipy.linalg.solve(L, b.T)
        z = scipy.linalg.solve(U, y).T
        v = z - v

    snap4 = v

    return (snap0, snap1, snap2, snap3, snap4)


# Abimeetod, mille abil saame edaspidi tulemusi graafikule kanda

def draw_snaps(snaps_arr):
    plt.figure(figsize=(20, 20))
    plt.rcParams['font.size'] = 14

    rows = int(len(snaps_arr)/2) + len(snaps_arr) % 2

    plt.figure(1)
    for j, snaps in enumerate(snaps_arr):
        plt.subplot(rows, 2, j+1)
        plt.title(snaps[1])
        for i, snap in enumerate(snaps[0]):
            plt.plot(snap, label="snap%d" % i)
        plt.legend(loc='best')
    plt.show()


# Soojuse levik erinevate difusioonikordajatega

f = lambda x: -3*np.sin(x)

left_func = lambda x: x**2
right_func = lambda x: x**2

snaps_arr1 = [
    (get_snaps(0, f, left_func, right_func), "$k = 0$"),
    (get_snaps(1, f, left_func, right_func), "$k = 1$"),
    (get_snaps(10, f, left_func, right_func), "$k = 10$"),
    (get_snaps(20, f, left_func, right_func), "$k = 20$"),
    (get_snaps(50, f, left_func, right_func), "$k = 50$"),
]

draw_snaps(snaps_arr1)


# Soojuse levik erinevate algjaotuste korral

left_func = lambda x: x**2 / 2
right_func = lambda x: x**2 / 2

snaps_arr2 = [
    (get_snaps(5, lambda x: -50*np.sin(x), left_func, right_func), "$f(x) = -50 \sin(x)$"),
    (get_snaps(5, lambda x: -10*np.sin(x), left_func, right_func), "$f(x) = -10 \sin(x)$"),
    (get_snaps(5, lambda x: np.sin(x), left_func, right_func), "$f(x) = \sin(x)$"),
    (get_snaps(5, lambda x: 10*np.sin(x), left_func, right_func), "$f(x) = 10 \sin(x)$"),
    (get_snaps(5, lambda x: 50*np.sin(x), left_func, right_func), "$f(x) = 50 \sin(x)$"),
]

draw_snaps(snaps_arr2)


# Soojuse levik erinevate vasakpoolsete rajatingimuste korral

right_func = lambda x: x**2 / 2

snaps_arr3 = [
    (get_snaps(5, np.sin, lambda x: x**2 / 2, right_func),
        r"$left(x) = \frac{x^2}{2}$"),
    (get_snaps(5, np.sin, lambda x: x**3, right_func),
        "$left(x) = x^3$"),
    (get_snaps(5, np.sin, lambda x: x**2 - x, right_func),
        "$left(x) = x^2 - x$"),
    (get_snaps(5, np.sin, lambda x: np.sqrt(np.abs(x)), right_func),
        "$left(x) = \sqrt{|x|}$"),
    (get_snaps(5, np.sin, np.exp, right_func),
        "$left(x) = e^x$"),
]

draw_snaps(snaps_arr3)


# Soojuse levik erinevate parempoolsete rajatingimuste korral

left_func = lambda x: x**2 / 2

snaps_arr4 = [
    (get_snaps(5, np.sin, left_func, np.abs), r"$right(x) = |x|$"),
    (get_snaps(5, np.sin, left_func, lambda x: np.cos(np.exp(x))), r"$right(x) = \cos e^x$"),
    (get_snaps(5, np.sin, left_func, lambda x: x), "$right(x) = x$"),
    (get_snaps(5, np.sin, left_func, np.tan), r"$right(x) = \tan x$"),
    (get_snaps(5, np.sin, left_func, lambda x: x + 100), r"$right(x) = x + 100$"),
]

draw_snaps(snaps_arr4)


# Soojuse levik erinevate vabaliikmete q korral

snaps_arr5 = [
    (get_snaps2(5, np.sin, np.abs), r"$q(x) = |x|$"),
    (get_snaps2(5, np.sin, lambda x: np.cos(np.exp(x))), r"$q(x) = \cos e^x$"),
    (get_snaps2(5, np.sin, lambda x: x), "$q(x) = x$"),
    (get_snaps2(5, np.sin, np.tan), r"$q(x) = \tan x$"),
    (get_snaps2(5, np.sin, lambda x: x + 100), r"$q(x) = x + 100$"),
]

draw_snaps(snaps_arr5)
