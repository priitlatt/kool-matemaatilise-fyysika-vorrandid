# coding: utf-8

# MTMM.00.209 Matemaatilise füüsika võrrandid
# 3. kodutöö - Priit Lätt

# Ülesande püstituse ja valemite tekke kohta võib vaadata selgitusi veebilehelt
# http://nbviewer.ipython.org/github/priitlatt/matemaatilise-fyysika-vorrandid/
# blob/master/yl3/mfv_yl3_priit_latt.ipynb

# Impordime edaspidises vajaminevad moodulid `numpy` ja `matplotlib`
# vastavalt numbriliste operatsioonide sooritamiseks ja animatsiooni
# koostamiseks.

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

# Defineerime klassi `Wave2D`, mis vastavalt etteantud parameetritele tekitab
# piirkonna $\Omega$ esituse gridina, kus saab lainevõrrandeid lahendada.


class Wave2D(object):

    def __init__(self, height, width, T, nx, ny, nt, c,
                 center_x=0.5, center_y=0.5):
        """
        Laine konstruktor. Inistialiseerib etteantud andmete põhjal
        ristküliku, kus on võimalik arvutada suvaliste funktsioonide poolt
        määratud lainevõrrandite levimist.

        Parameetrid
        -----------
        height : int
            Laine levimise ruumi kõrgus
        width : int
            Laine levimise ruumi laius
        T : float
            Lõppaeg
        nx : int
            Ruumis x-teljel olevate punktide arv
        ny : int
            Ruumis y-teljel olevate punktide arv
        nt : int
            Ajasammude arv
        c : float
            Lain-e levimise kiirus
        """

        if not (0 <= center_x <= 1 and 0 <= center_y <= 1):
            raise ValueError("center_x and center_y must be from [0, 1]")

        self.nx, self.ny, self.nt = nx, ny, nt

        # Konstrueerime ruumi määravad vektorid:
        # x = [-0.5*width, 0.5*width]
        # y = [-0.5*height, 0.5*height]
        self.x = np.linspace((center_x-1)*width, center_x*width, nx)
        self.y = np.linspace((center_y-1)*height, center_y*height, ny)
        # Koostame ajavektori t = [0, T], kus T tähistab etteantud lõppaega
        self.t = np.linspace(0, T, nt+1)

        # Fikseerime x-teljel, y-teljel ja ajavektoril olevate
        # punktikeste vahed dx, dy ja dt
        self.dx = self.x[1]-self.x[0]
        self.dy = self.y[1]-self.y[0]
        self.dt = self.t[1]-self.t[0]

        # Moodustame x-telje ja y-telje abil 2-mõõtmelise gridi,
        # mis määrab meie ruumi
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        # Arvutame (gamma_x)^2 ja (gamma_y)^2
        self.gx2 = c*self.dt / self.dx  # gamma_x^2
        self.gy2 = c*self.dt / self.dy  # gamma_y^2
        # Arvutame (gamma_x)^2 ja (gamma_y)^2 põhjal gamma
        self.gamma = 2 * (1 - self.gx2 - self.gy2)

    def solve(self, f_fun, g_fun):
        """
        Parameetrid
        -----------
        f_fun : function
            Kahemuutuja funktsioon, mis töötab vektoritel
        g_fun : function
            Kahemuutuja funktsioon, mis töötab vektoritel
        """

        # Arvutame ruumis ette antud funktsioonide f_fun ja g_fun väärtused
        f = f_fun(self.xx, self.yy)
        g = g_fun(self.xx, self.yy)

        # Initsialiseerime lahendi u nullvektoriga
        u = np.zeros((self.ny, self.nx, self.nt+1))
        # Nõuame algtingimuse täidetust ajahetkel t = 0
        u[:, :, 0] = f

        # arvutame esimese sammu
        u[:, :, 1] = 0.5*self.gamma*f + g*self.dt
        u[1:-1, 1:-1, 1] += 0.5*self.gx2*(f[1:-1, 2:]+f[1:-1, :-2])
        u[1:-1, 1:-1, 1] += 0.5*self.gy2*(f[:-2, 1:-1]+f[2:, 1:-1])

        # Arvutame kõik ülejäänud sammud
        for k in range(1, self.nt):
            # kõik punktid sisaldavad neid osasid
            u[:, :, k+1] = self.gamma*u[:, :, k] - u[:, :, k-1]

            # sisepunktid
            u[1:-1, 1:-1, k+1] += self.gx2*(u[1:-1, 2:, k]+u[1:-1, :-2, k]) + \
                self.gy2*(u[2:, 1:-1, k]+u[:-2, 1:-1, k])

            # ülemine raja
            u[0, 1:-1, k+1] += 2*self.gy2*u[1, 1:-1, k] + \
                self.gx2*(u[0, 2:, k]+u[0, :-2, k])

            # parem raja
            u[1:-1, -1, k+1] += 2*self.gx2*u[1:-1, -2, k] + \
                self.gy2*(u[2:, -1, k]+u[:-2, -1, k])

            # alumine raja
            u[-1, 1:-1, k+1] += 2*self.gy2*u[-2, 1:-1, k] + \
                self.gx2*(u[-1, 2:, k]+u[-1, :-2, k])

            # vasak raja
            u[1:-1, 0, k+1] += 2*self.gx2*u[1:-1, 1, k] + \
                self.gy2*(u[2:, 0, k]+u[:-2, 0, k])

            # parem ülemine nurk
            u[0, -1, k+1] += 2*self.gx2*u[0, -2, k] + 2*self.gy2*u[1, -1, k]

            # parem alumine nurk
            u[-1, -1, k+1] += 2*self.gx2*u[-1, -2, k] + 2*self.gy2*u[-2, -1, k]

            # vasak alumine nurk
            u[-1, 0, k+1] += 2*self.gx2*u[-1, 1, k] + 2*self.gy2*u[-2, 0, k]

            # vasak ülemine nurk
            u[0, 0, k+1] += 2*self.gx2*u[0, 1, k] + 2*self.gy2*u[1, 0, k]

        # Tagastame lahendi u
        return u


def animate_wave(wave, u):
    """ Abifunktsioon, mis genereerib lahendist animatsiooni. """
    x = wave.x
    y = wave.y

    frames = []
    fig = plt.figure(1, (10, 5))

    for k in range(wave.nt+1):
        frame = plt.imshow(u[:, :, k], extent=[x[0], x[-1], y[0], y[-1]])
        frames.append([frame])

    return animation.ArtistAnimation(fig, frames, interval=100,
                                     blit=True, repeat_delay=1000)


# Fikseerime järgnevalt mõned piirkonda $\Omega$ kirjeldavad karakteristikud
# ja võrrandi lahendamiseks vajalikud parameetrid.

# Määrame ruumis lõpphetkeks aja $T = 0.5$.
T = 0.5

# Määrame $\Omega$ kui ristküliku laiuseks $4$ ja kõrguseks $4$.
width = 4
height = 2

# Määrame laine levimise kiiruseks $c = 10$.
c = 10

# Määrame ajasammude arvuks $700$ ja $x$-telje ning $y$-telje suunaliste
# punktide arvudeks vastavalt $200$ ning $100$.
nt = 700
nx = 200
ny = 100


# Loome eelnevalt defineeritud parameetritele vastavalt kaks laine levimise
# ruumi kasutades klassi `Wave2D`, kus
# * esimeses hakkab laine levima ruumi keskelt ning
# * teises ruumi vasakust ülemisest nurgast.
wave_center = Wave2D(height, width, T, nx, ny, nt, c)
wave_top_left = Wave2D(height, width, T, nx, ny, nt, c, center_x=1, center_y=1)


# Määrame lainefunktsioonile rajatingumused
# f(x, y) = e^(-10(x^2 + y^2)) ja g(x, y) = sin(x) + cos(y).
f = lambda x, y: np.exp(-10*(x**2+y**2))
g = lambda x, y: np.sin(x) + np.cos(y)

# Leiame mõlemas ruumis nende rajatingimustega $f$ ja $g$ määratud
# lainevõrrandite lahendid $u_1$ ja $u_2$.
u1 = wave_center.solve(f, g)
u2 = wave_top_left.solve(f, g)

# Koostame esimesest lahendist $u_1$ animatsiooni.
anim1 = animate_wave(wave_center, u1)
plt.show()

# Koostame teisest lahendist $u_2$ animatsiooni.
anim2 = animate_wave(wave_top_left, u2)
plt.show()

# Initsialiseerime uue ristküliku ning vaatame seal laine levimist, kui
# rajatingimused on antud funktsioonidega
# f(x, y) = x + y ja g(x, y) = x \cdot y
# ning
# f(x, y) = |x|^y ja g(x, y) = e^(x - y).
wave = Wave2D(2, 4, 0.5, 100, 50, 700, 3)

f = lambda x, y: x + y
g = lambda x, y: x*y
u3 = wave.solve(f, g)

anim3 = animate_wave(wave, u3)
plt.show()

f = lambda x, y: np.abs(x)**y
g = lambda x, y: np.exp(x-y)
u4 = wave.solve(f, g)

anim4 = animate_wave(wave, u4)
plt.show()
