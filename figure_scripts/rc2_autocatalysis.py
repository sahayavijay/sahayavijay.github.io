"""Homepage figure rc2: growth or decay of a linear autocatalytic cycle in an open system.

Network: X1 -> X2 -> ... -> Xm -> 2 X1, all with rate k, and every species lost at rate d.
Exact growth rate: lambda = k (2^(1/m) - 1) - d. Rates are illustrative, in units of k = 1.
"""
import sys
from scipy.integrate import solve_ivp
from matplotlib.patches import Ellipse
from figure_style import *

M_SPECIES, K = 6, 1.0


def matrix(m, k, d):
    A = -(k + d) * np.eye(m)
    for i in range(m - 1):
        A[i + 1, i] = k
    A[0, m - 1] = 2 * k
    return A


def exact(m, k, d):
    return k * (2 ** (1 / m) - 1) - d


def main(outdir):
    dstar = exact(M_SPECIES, K, 0)
    for m, k, d in ((3, 1, 0.1), (6, 1, 0.05), (6, 2, 0.3), (12, 0.5, 0.01)):
        assert abs(np.linalg.eigvals(matrix(m, k, d)).real.max() - exact(m, k, d)) < 1e-10
    print(f"check passed: eigenvalue = closed form. d* = {dstar:.5f}")

    fig, axa, axb = new_figure()
    # Panel a: the cycle in perspective, sphere size from the dominant eigenvector
    w, v = np.linalg.eig(matrix(M_SPECIES, K, 0.5 * dstar))
    u = np.abs(v[:, w.real.argmax()].real); u /= u.max()
    print("dominant eigenvector, X1 to X6:", np.round(u, 3))
    t = np.pi / 2 - np.arange(M_SPECIES) * 2 * np.pi / M_SPECIES
    x, y = np.cos(t), np.sin(t)
    tilt, cam = np.radians(50), 6.0
    scale = cam / (cam + y * np.sin(tilt))
    p = np.column_stack((x * scale, y * np.cos(tilt) * scale))
    for i in range(M_SPECIES):
        axa.add_patch(Ellipse((p[i, 0], p[i, 1] - 0.27 * scale[i]), 0.48 * scale[i], 0.3 * scale[i], fc=SHADOW, ec="none", alpha=0.32, zorder=0))
    rad = (0.09 + 0.13 * u) * scale
    for i in range(M_SPECIES):
        j = (i + 1) % M_SPECIES
        last = i == M_SPECIES - 1
        d = p[j] - p[i]; n = d / np.linalg.norm(d)
        axa.annotate("", xy=p[j] - n * (rad[j] + 0.015), xytext=p[i] + n * (rad[i] + 0.015), zorder=1,
                     arrowprops=dict(arrowstyle="-|>,head_width=0.16,head_length=0.32", lw=1.5 if last else 0.9,
                                     color=ink(RED) if last else STICK, shrinkA=0, shrinkB=0))
    for i in np.argsort(-y):
        axa.add_patch(plt.Circle(p[i], rad[i], fc=RED if i == 0 else BLUE, ec="none", zorder=3))
    for i, name in enumerate([f"X$_{n}$" for n in range(1, 7)]):
        side = np.sign(x[i]) if abs(x[i]) > 0.1 else 0
        o = p[i] + np.array([side * (rad[i] + 0.05), 0.02 if side else np.sign(y[i]) * (rad[i] + 0.08)])
        axa.text(*o, name, fontsize=6, ha={1: "left", -1: "right", 0: "center"}[int(side)], va="center", color="#222222", zorder=4)
    mid = (p[-1] + p[0]) / 2
    axa.text(mid[0] - 0.08, mid[1] + 0.10, "×2", color=ink(RED), fontsize=7, weight="bold", ha="right")
    axa.set_aspect("equal"); axa.axis("off")
    axa.set_xlim(-1.45, 1.45); axa.set_ylim(p[:, 1].min() - 0.5, p[:, 1].max() + 0.3)
    key(fig, [(5, RED, 1.2, "Autocatalyst"), (24, BLUE, 1.2, "Intermediate")])

    # Panel b: ODE total amount against time, and the exact long-time slope
    t_eval = np.linspace(0, 100, 21)
    x0 = np.eye(M_SPECIES)[0]
    for ratio, c, label, xtxt, ytxt in ((0.5, GREEN, "$d = 0.5\\,d^*$", 56, 4.0), (1.0, DEEP, "$d = d^*$", 62, 0.2), (1.5, RED, "$d = 1.5\\,d^*$", 6, 3e-3)):
        d = ratio * dstar
        A = matrix(M_SPECIES, K, d)
        sol = solve_ivp(lambda _, z: A @ z, (0, 100), x0, t_eval=t_eval, method="DOP853", rtol=1e-11, atol=1e-14)
        N = sol.y.sum(axis=0)
        lam = exact(M_SPECIES, K, d)
        fit = np.polyfit(t_eval[-6:], np.log(N[-6:]), 1)[0]
        print(f"d/d*={ratio}: lambda exact {lam:+.6f}, ODE slope {fit:+.6f}, N(100)={N[-1]:.4g}")
        assert abs(fit - lam) < 1e-5
        axb.semilogy(t_eval, N[-1] * np.exp(lam * (t_eval - 100)), "--", color=SLOPE, lw=0.8, zorder=2)
        axb.scatter(t_eval, N, s=13, c=c, ec="none", zorder=3, clip_on=False)
        axb.text(xtxt, ytxt, label, color=ink(c), fontsize=6.5, ha="left", va="center")
    axb.set_xlim(0, 100); axb.set_ylim(1e-4, 1e3)
    axb.set_xlabel(r"Time, $kt$", labelpad=1.5)
    axb.set_ylabel(r"Total amount, $N$", labelpad=1.5)
    axb.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=9))
    axb.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    save(fig, Path(outdir) / "rc2_autocatalysis.svg")


if __name__ == "__main__":
    main(sys.argv[1])
