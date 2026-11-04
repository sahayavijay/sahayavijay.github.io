"""Homepage figure rc5: the validation step, shown on real numerical tests.

Test: the growth rate of a linear autocatalytic cycle is computed by time integration and checked
against the exact value lambda = k (2^(1/m) - 1) - d. A result passes if its relative error is below 1e-6.
"""
import sys
from scipy.integrate import solve_ivp
from figure_style import *
from rc2_autocatalysis import matrix, exact

THRESHOLD = 1e-6


def growth_rate(m, k, d, rtol):
    A = matrix(m, k, d)
    T = 160 / k
    t = np.linspace(0.8 * T, T, 9)
    sol = solve_ivp(lambda _, z: A @ z, (0, T), np.eye(m)[0], t_eval=t, method="RK45", rtol=rtol, atol=1e-120)
    return np.polyfit(t, np.log(sol.y.sum(axis=0)), 1)[0]


def main(outdir):
    ms = np.arange(3, 9)
    tols = 10.0 ** np.arange(-2, -11, -2)
    k, d = 1.0, 0.05
    err = np.array([[abs(growth_rate(m, k, d, tol) / exact(m, k, d) - 1) for tol in tols] for m in ms])
    print("relative error, rows m=3..8, columns rtol=1e-2..1e-10\n", np.array2string(err, precision=1))
    print(f"pass {int((err < THRESHOLD).sum())} of {err.size}")

    fig, axa, axb = new_figure(panel_a_3d=True)
    style_3d(axa, 20, -58, r"log$_{10}$ tolerance", r"Cycle length, $m$", r"log$_{10}$ error")
    X, Y = np.meshgrid(np.log10(tols), ms)
    Z = np.log10(np.maximum(err, 1e-14))
    fl = -14
    xx, yy = np.meshgrid([-11, -1], [2.5, 8.5])
    axa.plot_surface(xx, yy, np.full_like(xx, np.log10(THRESHOLD), dtype=float), color="#d7dbe3", alpha=0.5, linewidth=0, zorder=1, shade=False)
    for x, y, z in zip(X.ravel(), Y.ravel(), Z.ravel()):
        axa.plot([x, x], [y, y], [fl, z], color="#a4a9b2", lw=0.4, zorder=2)
    axa.scatter(X.ravel(), Y.ravel(), Z.ravel(), s=13, c=[GREEN if e < THRESHOLD else RED for e in err.ravel()], ec="none", depthshade=False, zorder=3)
    axa.set_xlim(-11, -1); axa.set_ylim(2.5, 8.5); axa.set_zlim(fl, 0)
    axa.set_xticks([-10, -6, -2]); axa.set_yticks([3, 8]); axa.set_zticks([-12, -6, 0])
    axa.set_box_aspect((1.2, 1, 0.8), zoom=0.94)
    key(fig, [(5, GREEN, 1.2, "Pass"), (18, RED, 1.2, r"Fail, relative error above $10^{-6}$")])

    # Panel b: parity of the computed and exact growth rate for random networks
    rng = np.random.default_rng(20261104)
    cases = [(int(rng.integers(3, 9)), rng.uniform(0.5, 2.0), rng.uniform(0.0, 0.5)) for _ in range(40)]
    ref = np.array([exact(*c) for c in cases])
    num = np.array([growth_rate(*c, 1e-10) for c in cases])
    print(f"parity: {len(cases)} random networks, max |num - exact| = {np.abs(num - ref).max():.2e}, lambda range {ref.min():.3f} to {ref.max():.3f}")
    assert np.abs(num - ref).max() < 1e-6
    lim = (-0.6, 0.6)
    assert ref.min() > lim[0] and ref.max() < lim[1]
    axb.plot(lim, lim, "--", color=SLOPE, lw=0.9, zorder=2)
    axb.scatter(ref, num, s=14, c=DEEP, ec="none", zorder=3)
    axb.set_xlim(lim); axb.set_ylim(lim); axb.set_aspect("equal", adjustable="box")
    ticks = [-0.5, 0, 0.5]
    labels = ["−0.5", "0", "0.5"]
    axb.set_xticks(ticks, labels); axb.set_yticks(ticks, labels)
    axb.set_xlabel("Exact growth rate", labelpad=1.5)
    axb.set_ylabel("Computed growth rate", labelpad=0.5, fontsize=7.5)
    save(fig, Path(outdir) / "rc5_ai_tools.svg")


if __name__ == "__main__":
    main(sys.argv[1])
