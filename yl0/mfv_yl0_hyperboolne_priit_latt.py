# -*- coding: utf-8 -*-

"""
Hüperboolne ODV:
\[ \frac{\delta^2 u}{\delta t^2} - \beta^2 \Delta u = 0. \]

Lahendid:
\[ u(x, t) = \sin(\pi x - \pi t), \]
\[ u(x, t) = \cos(\pi x - \pi t), \]
\[ u(x, t) = e^{-i(\pi x - \pi t)}, \]
kus $x \in [0, 1]$ ja $y \in [0, 1]$.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

exp = np.exp
sin = np.sin
cos = np.cos
pi = np.pi
e = np.e

# loome uue graafiku
fig3 = plt.figure(figsize=(20, 10))
# seame graafiku teljestiku kolmemõõtmeliseks
ax3 = fig3.gca(projection='3d')

# anname imaginaarühikule standardse kuju
i = 1j
# koostame vektori (0, 100] sammuga 0.5 (lõpmatusse ei saa me nagu nii minna)
dom3 = np.arange(0, 1, 0.01)
# väärtustame x ja t määramispiirkonnal tehtud otsekorrutisest,
# mida on vaja 3d joonise jaoks
x3, t3 = np.meshgrid(dom3, dom3)

# defineerime ODV lahendi funktsioonid
f3_1 = lambda x, t: sin(pi*x - pi*t)
f3_2 = lambda x, t: cos(pi*x - pi*t)
f3_3 = lambda x, t: exp(-i*(pi*x - pi*t))

# lisame graafikule lahendite poolt määratud pinnad
ax3.plot_surface(x3, t3, f3_1(x3, t3), color='r')
ax3.plot_surface(x3, t3, f3_2(x3, t3), color='g')
ax3.plot_surface(x3, t3, np.real(f3_3(x3, t3)), color='b')

# kuvame graafiku
plt.show()
