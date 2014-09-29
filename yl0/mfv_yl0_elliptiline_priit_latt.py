# -*- coding: utf-8 -*-

"""
Elliptiline ODV:
\[ \Delta u = 0. \]

Lahend:
\[ u(x, y) = (1 - x)(1 + x)(1 - y)(1 + y), \]
kus $x \in [-1, 1]$ ja $y \in [-1,1]$.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# loome uue graafiku
fig1 = plt.figure(figsize=(20, 10))
# seame graafiku teljestiku kolmemõõtmeliseks
ax1 = fig1.gca(projection='3d')

# koostame vektori [-1, 1] sammuga 0.01
dom1 = np.arange(-1, 1, 0.01)
# väärtustame x ja y määramispiirkonnal tehtud otsekorrutisest,
# mida on vaja 3d joonise jaoks
x1, y1 = np.meshgrid(dom1, dom1)

# defineerime ODV lahendi funktsiooni
f1 = lambda x, y: (1-x) * (1+x) * (1-y) * (1+y)

# lisame graafikule lahendi poolt määratud pinna
ax1.plot_surface(x1, y1, f1(x1, y1), color='b')

# kuvame graafiku
plt.show()
