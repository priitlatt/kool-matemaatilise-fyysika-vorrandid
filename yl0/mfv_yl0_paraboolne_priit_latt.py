# -*- coding: utf-8 -*-

"""
Paraboolne ODV:
\[ \frac{\delta u}{\delta t} - \beta^2 \Delta u = 0. \]

Lahend:
\[ u(x, t) = \frac{e^{-\frac{x^2}{4 \beta^2 t}}}{\sqrt{4 \beta^2 \pi t}}, \]
kus $x \in [0, \infty]$ ja $y \in [0, \infty]$.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pi = np.pi
sqrt = np.sqrt
exp = np.exp

# loome uue graafiku
fig2 = plt.figure(figsize=(20, 10))
# seame graafiku teljestiku kolmemõõtmeliseks
ax2 = fig2.gca(projection='3d')

# koostame vektori (0, 100] sammuga 0.5 (lõpmatusse ei saa me nagu nii minna)
dom2 = np.arange(0.1, 100, 0.5)
# väärtustame x ja t määramispiirkonnal tehtud otsekorrutisest,
# mida on vaja 3d joonise jaoks
x2, t2 = np.meshgrid(dom2, dom2)
# anname parameetrile \beta väärtuseks näiteks 5
beta = 5

# defineerime ODV lahendi funktsiooni
f2 = lambda x, t, beta: exp(-(x**2)/(4*(beta**2)*t)) / sqrt(4*(beta**2)*pi*t)

# lisame graafikule lahendi poolt määratud pinna
ax2.plot_surface(x2, t2, f2(x2, t2, beta), color='g')

# kuvame graafiku
plt.show()
