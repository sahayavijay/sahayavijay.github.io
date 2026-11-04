"""Homepage figure rc4: impedance simulated in the time domain for a Volmer-Heyrovsky reaction.

Textbook mechanism with illustrative rate constants, not fitted to any experiment.
Volmer H+ + e -> H*, Heyrovsky H* + H+ + e -> H2. Coverage ODE: Gamma dtheta/dt = r1 - r2.
Each frequency: a 1 mV sine is applied, the ODE is integrated, and Z comes from a Fourier analysis.
This is compared with the small-signal (linearised) impedance.
"""
import sys
from scipy.integrate import solve_ivp
from figure_style import *

F, f, A = 96485.0, 38.92, 0.5
GAMMA, RS, CDL = 2e-9, 5.0, 20e-6            # mol/cm2, ohm cm2, F/cm2
K1, K1R, K2, K2R = 1e-9, 1e-8, 2e-10, 1e-14  # mol/(cm2 s)
ETA0, DE = -0.10, 0.001


def rates(theta, eta):
    c, a = np.exp(-A * f * eta), np.exp((1 - A) * f * eta)
    return K1 * (1 - theta) * c - K1R * theta * a, K2 * theta * c - K2R * (1 - theta) * a


def net(theta, eta):
    r1, r2 = rates(theta, eta)
    return r1 - r2


def current(theta, eta):
    r1, r2 = rates(theta, eta)
    return -F * (r1 + r2)


def steady():
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if net(mid, ETA0) > 0 else (lo, mid)
    return (lo + hi) / 2


def linear_zf(w, th):
    h = 1e-6
    dtheta = lambda g: (g(th + h, ETA0) - g(th - h, ETA0)) / (2 * h)
    deta = lambda g: (g(th, ETA0 + h) - g(th, ETA0 - h)) / (2 * h)
    th_e = deta(net) / (1j * w * GAMMA - dtheta(net))
    return 1 / (deta(current) + dtheta(current) * th_e)


def time_domain_zf(fr, th, tau):
    w = 2 * np.pi * fr
    t0 = round(min(max(8 * tau, 3 / fr), 400 / fr) * fr) / fr
    window = 4 / fr
    eta = lambda t: ETA0 + DE * np.sin(w * t)
    sol = solve_ivp(lambda t, y: [net(y[0], eta(t)) / GAMMA], (0, t0 + window), [th],
                    method="LSODA", rtol=1e-10, atol=1e-13, max_step=1 / (40 * fr), dense_output=True)
    t = t0 + np.arange(1600) * window / 1600
    j = current(sol.sol(t)[0], eta(t))
    # Fit the sine, cosine and a slow drift together, so removing the drift does not disturb the signal
    basis = np.column_stack((np.sin(w * t), np.cos(w * t), np.ones_like(t), t - t.mean()))
    a, b, *_ = np.linalg.lstsq(basis, j, rcond=None)[0]
    return DE / (a + 1j * b)


def total(zf, w):
    return RS + 1 / (1j * w * CDL + 1 / zf)


def main(outdir):
    th = steady()
    assert abs(net(th, ETA0)) < 1e-20
    h = 1e-6
    tau = GAMMA / abs((net(th + h, ETA0) - net(th - h, ETA0)) / (2 * h))
    print(f"steady coverage = {th:.5f}, j = {current(th, ETA0)*1e3:.4f} mA/cm2, coverage relaxation time = {tau:.4f} s")
    freqs = np.geomspace(1e-2, 1e4, 25)
    Zf = np.array([time_domain_zf(fr, th, tau) for fr in freqs])
    Zn = total(Zf, 2 * np.pi * freqs)
    Za = total(linear_zf(2 * np.pi * freqs, th), 2 * np.pi * freqs)
    rel = np.abs(Zn - Za) / np.abs(Za)
    print(f"|Z_time - Z_linear|/|Z_linear|: median {np.median(rel):.2e}, max {rel.max():.2e} at {freqs[rel.argmax()]:.3g} Hz")
    print(f"Z(lowest f) = {Zn[0]:.2f}, Z(highest f) = {Zn[-1]:.2f}")
    assert rel.max() < 0.01

    fig, axa, axb = new_figure(panel_a_3d=True)
    style_3d(axa, 22, -62, r"$Z'$", r"log$_{10}$ $f$", r"$-Z''$")
    lf = np.log10(freqs)
    fine = np.geomspace(freqs[0], freqs[-1], 300)
    Zl = total(linear_zf(2 * np.pi * fine, th), 2 * np.pi * fine)
    axa.plot(Zl.real, np.log10(fine), -Zl.imag, color=ink(STICK), lw=0.8, zorder=2)
    axa.plot(Zl.real, np.log10(fine), 0 * fine, color="#b8bdc6", lw=0.8, zorder=1)
    axa.scatter(Zn.real, lf, -Zn.imag, s=14, c=DEEP, ec="none", depthshade=False, zorder=3)
    top = 1.08 * (-Zn.imag).max()
    span = 1.05 * Zn.real.max()
    axa.set_xlim(0, span); axa.set_ylim(-2, 4); axa.set_zlim(0, top)
    axa.set_yticks([-2, 1, 4]); axa.set_xticks([0, round(span / 2, -2)]); axa.set_zticks([0, round(top / 2, -1)])
    axa.set_box_aspect((1.25, 1, 0.75), zoom=0.98)
    key(fig, [(5, DEEP, 1.2, "Time-domain simulation")])

    axb.plot(Zl.real, -Zl.imag, "-", color=SLOPE, lw=0.9, zorder=2)
    axb.scatter(Zn.real, -Zn.imag, s=14, c=DEEP, ec="none", zorder=3, clip_on=False)
    lim = 1.08 * Zn.real.max()
    axb.set_xlim(0, lim); axb.set_ylim(0, lim); axb.set_aspect("equal", adjustable="box")
    mag = 10 ** np.floor(np.log10(lim))
    ticks = np.arange(0, lim, mag if lim / mag < 3 else 2 * mag)
    axb.set_xticks(ticks); axb.set_yticks(ticks)
    axb.set_xlabel(r"$Z'$ (Ω cm²)", labelpad=1.5)
    axb.set_ylabel(r"$-Z''$ (Ω cm²)", labelpad=1.5)
    axb.text(0.97, 0.90, "Linearised model", color=ink(SLOPE), fontsize=6, ha="right", transform=axb.transAxes)
    save(fig, Path(outdir) / "rc4_eis.svg")


if __name__ == "__main__":
    main(sys.argv[1])
