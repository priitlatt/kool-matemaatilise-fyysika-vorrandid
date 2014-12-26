
# coding: utf-8

# MTMM.00.209 Matemaatilise füüsika võrrandid
#
# 4. kodutöö - Priit Lätt

# Ülesande püstitus

# Lahendame järgmise **Poissoni** võrrandi:
# \begin{align*}
#     - \Delta u(x, y) &= f(x, y), \quad x, y \in \Omega, \\
#     u(x, y) &= g(x, y), \quad x, y \in \Gamma_D, \\
#     \hat{n} \cdot \nabla u(x, y) &= h(x, y), \quad x, y \in \Gamma_N,
# \end{align*}
# kus määramispiirkond $\Omega = [-1, 1]^2$ ja piirkonna $\Omega$ raja on
# esitatud tükiti $\partial \Omega = \Gamma_D + \Gamma_N$. Siin on
# $\Gamma_D$ raja osa, kus kesktivad *Dirichlet* rajatingimused ning
# $\Gamma_N$ on raja osa, kus kehtivad *Neumanni* tingimused.

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import solve

import orthopoly as op
from pseudospectral import diffmat


def poisson_2d(n, f, g, h, d):
    """
    Poissioni võrrandi (1) lahendamine ruudul [-1, 1]^2
    koos rajatingimustega:
    -\Delta u = f(x, y), kus -1 < x, y < 1 (1)
    koos Dirichlet või Neumann tingimustega, mis on
    määratud indikaatorfunktsiooniga d(x, y).

    Sisendid:
    n - testpolünoomi järk mõlemas suunas
    f - algtingimuse (1) parem pool
    g - Dirichlet andmed
    h - Neumann andmed
    d - indikaatortingimus, millal peaks kehtima Dirichlet' tingimus

    Väljundid:
    xx, yy - grid
    U - Lahend grid'il
    """

    alpha, beta = op.rec_jacobi(n, 0, 0)   # Legendre rekursiivsed koefitsendid
    x, w = op.lobatto(alpha, beta, -1, 1)  # Legendre-Gauss-Lobatto kvadratuur
    D = diffmat(x)                         # Pseudospectral diferentseerimise
                                           # maatriks
    M = np.diag(w)                         # Ligikaudne kaalude maatriks
    K = np.dot(D.T, np.dot(M, D))          # Jäikuse maatriks
    xx, yy = np.meshgrid(x, x)             # Koostame x ja y telje põhjal gridi
    xf = xx.flatten()                      # Valime gridist veeru väärtused
    yf = yy.flatten()

    # Eraldame erinevat tüüpi punktide indeksid
    k = np.arange(1, n-1)
    dex = set(np.arange(n*n))
    bdex = np.hstack((0, k, n-1, n*k, (k+1)*n-1, n*(n-1), n*(n-1)+k, n*n-1))

    dbool = d(xf[bdex], yf[bdex]) > -1e-9  # Tagastab True Dirichlet punktidele
    ddex = set(bdex[dbool])                # Dirichlet punktide indeksid
    bdex = set(bdex)
    ndex = list(bdex.difference(ddex))     # Neumanni punktide indeksid
    udex = list(dex.difference(ddex))      # Tundmatute indeksid
    ddex = list(ddex)
    ndex = list(ndex)

    W = np.zeros((n, n))                   # Pinna kvadratuur
    W[0, :] = w
    W[:, 0] = w
    W[-1, :] = w
    W[:, -1] = w
    W = W.flatten()

    H = np.zeros(n*n)
    H[ndex] = h(xf[ndex], yf[ndex])        # Neumanni pinna andmed

    A = np.kron(K, M) + np.kron(M, K)      # Galerkini lähenduses -Delta

    F = np.kron(w, w)*f(xf, yf)            # Poisson võrrand
    F = F[udex]

    G = g(xf[ddex], yf[ddex])              # Dirichlet andmed
    Au = A[udex, :][:, udex]               # Taandame süsteemi tundmatutele
    F -= np.dot(A[udex, :][:, ddex], G)    # Kasutame PP Dirichlet andmeid
    F += H[udex]*W[udex]                   # Kasutame PP Neumanni andmeid

    u = np.zeros(n*n)
    u[ddex] = G                            # Sätime u teadaolevad väärtused
    u[udex] = solve(Au, F, sym_pos=True)   # Lahendame tundmatute suhtes

    U = np.reshape(u, (n, n))

    Uexact = g(xx, yy)
    return xx, yy, U, Uexact


def manufactured_solution():
    import sympy as sy
    from sympy.abc import x, y

    u = x*sy.exp(x-y) + y*sy.exp(x+y)
    ux = sy.diff(u, x)
    uy = sy.diff(u, y)
    f = -sy.diff(ux, x) - sy.diff(uy, y)

    F = sy.lambdify([x, y], f, "numpy")
    G = sy.lambdify([x, y], u, "numpy")

    # Kasutame ruudu ülemises ja alumises servas Dirichlet tingimusi
    d = (y+1)*(y-1)
    D = sy.lambdify([x, y], d, "numpy")

    # Neumanni tingimused
    h = ux*sy.sign(x)
    H = sy.lambdify([x, y], h, "numpy")

    return F, G, H, D


def show_results(n, xx, yy, U, Uexact):
    Uerr = U - Uexact
    max_err = max(abs(Uerr.flatten()))
    print('N = %d\nMaksimaalne viga = %e' % (n, max_err))

    fig = plt.figure(figsize=(20, 8))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(xx, yy, U, rstride=1, cstride=1, cmap=cm.jet,
                     linewidth=0.1, antialiased=True)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Lahendus')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(xx, yy, Uerr, rstride=1, cstride=1, cmap=cm.jet,
                     linewidth=0.1, antialiased=True)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Viga')

    plt.show()


if __name__ == '__main__':
    f, g, h, d = manufactured_solution()

    for n in (5, 10, 15, 20, 25, 50):
        xx, yy, U, Uexact = poisson_2d(n, f, g, h, d)
        show_results(n, xx, yy, U, Uexact)
