"""Homepage figure rc3: timescale decomposition of a linear three-species network.

Network: A <-> B <-> C <-> A with illustrative first-order rates. The rate matrix eigenvalues give
two relaxation timescales. Numerical ODE trajectories are compared with the eigenvalue solution.
"""
import sys
from scipy.integrate import solve_ivp
from figure_style import *

# RATES[i, j] is the rate constant from species j to species i
RATES = np.array([[0, 0.4, 0.05], [1.0, 0, 0.3], [0.02, 0.12, 0]])
K = RATES - np.diag(RATES.sum(axis=0))
STARTS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0, 0.5], [0, 0.5, 0.5]], float)


def main(outdir):
    w, v = np.linalg.eig(K)
    order = np.argsort(np.abs(w.real))
    w, v = w[order].real, v[:, order].real
    ss = v[:, 0] / v[:, 0].sum()
    assert abs(w[0]) < 1e-12 and np.abs(K @ ss).max() < 1e-12 and ss.min() > 0
    tau = -1 / w[1:]
    print(f"steady state A,B,C = {np.round(ss, 4)}, eigenvalues = {np.round(w, 5)}, timescales = {np.round(tau, 3)}")

    fig, axa, axb = new_figure(panel_a_3d=True)
    style_3d(axa, 24, 38, "A", "B", "C")
    tri = STARTS[:3]
    axa.plot_trisurf(tri[:, 0], tri[:, 1], tri[:, 2], color="#c6ccd8", alpha=0.55, linewidth=0, zorder=1, shade=False)
    axa.plot(*np.vstack((tri, tri[0])).T, color=STICK, lw=0.6, zorder=2)
    t_dots = np.geomspace(0.05, 120, 16)
    for s in STARTS:
        sol = solve_ivp(lambda _, x: K @ x, (0, 120), s, method="DOP853", rtol=1e-10, atol=1e-12, dense_output=True)
        line = sol.sol(np.geomspace(1e-3, 120, 200))
        assert np.abs(sol.y.sum(axis=0) - 1).max() < 1e-8, "mass conservation"
        axa.plot(*line, color=ink(BLUE), lw=0.8, zorder=3)
        axa.scatter(*sol.sol(t_dots), s=4, c=BLUE, ec="none", depthshade=False, zorder=4)
    axa.scatter(*STARTS.T, s=15, c=RED, ec="none", depthshade=False, zorder=5)
    axa.scatter(*ss, s=34, c=GREEN, ec="none", depthshade=False, zorder=6)
    axa.set_xlim(0, 1); axa.set_ylim(0, 1); axa.set_zlim(0, 1)
    axa.set_xticks([0, 1]); axa.set_yticks([0, 1]); axa.set_zticks([0, 1])
    axa.set_box_aspect((1, 1, 0.9), zoom=0.98)
    key(fig, [(5, RED, 1.2, "Start"), (17, BLUE, 0.8, "Trajectory"), (34, GREEN, 1.5, "Steady state")])

    # Panel b: distance from steady state, numerical dots against the eigenvalue solution
    x0 = STARTS[0]
    t = np.linspace(0, 24, 17)
    sol = solve_ivp(lambda _, x: K @ x, (0, 24), x0, t_eval=t, method="DOP853", rtol=1e-12, atol=1e-14)
    c = np.linalg.solve(v, x0)
    exact = v @ (c[:, None] * np.exp(w[:, None] * t))
    err = np.abs(sol.y - exact).max()
    print(f"max |ODE - eigenvalue solution| = {err:.2e}")
    assert err < 1e-9
    dist = np.linalg.norm(sol.y - ss[:, None], axis=0)
    slow = np.linalg.norm(c[1] * v[:, 1]) * np.exp(w[1] * t)
    fast = np.linalg.norm(c[2] * v[:, 2]) * np.exp(w[2] * t)
    axb.semilogy(t, slow, "--", color=SLOPE, lw=0.9, zorder=2)
    axb.semilogy(t, fast, ":", color=SLOPE, lw=0.9, zorder=2)
    axb.scatter(t, dist, s=16, c=DEEP, ec="none", zorder=3, clip_on=False)
    axb.text(11, 2.0e-1, f"τ$_1$ = {tau[0]:.1f}", color=ink(SLOPE), fontsize=6.5)
    axb.text(8.5, 2.5e-5, f"τ$_2$ = {tau[1]:.2f}", color=ink(SLOPE), fontsize=6.5)
    axb.set_xlim(0, 25); axb.set_ylim(1e-5, 3)
    axb.set_xlabel("Time", labelpad=1.5)
    axb.set_ylabel("Distance from steady state", labelpad=1.5, fontsize=7)
    axb.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=8))
    axb.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    save(fig, Path(outdir) / "rc3_modelling.svg")


if __name__ == "__main__":
    main(sys.argv[1])
