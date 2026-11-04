"""Homepage Skills figures. Each one is the output of a real calculation, with its own check.

sk1 Programming: Gillespie stochastic simulation of a birth-death process.
sk2 Scientific software: Mueller-Brown surface, gradient derived symbolically with SymPy.
sk3 Research methods: Michaelis-Menten reaction network, ODE against the Lambert W solution.
sk4 Software I built: real PyTIE_D output, read from pytie_7agnr_output.txt.
sk5 Certifications: a small neural network trained with NumPy on generated data.
sk6 Languages: the Tamil and English alphabets, generated from Unicode code points.
"""
import re
import sys
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, root
from scipy.special import lambertw
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from figure_style import *

HERE = Path(__file__).resolve().parent
RGB = LinearSegmentedColormap.from_list("rgb", ["#0f2f7a", "#12703a", "#b8a01c", "#b01c1c"])


def single_axes(rect=(0.125, 0.2, 0.84, 0.74)):
    fig, axa, axb = new_figure()
    axa.remove(); axb.remove()
    for t in list(fig.texts):
        t.remove()
    ax = fig.add_axes(rect)
    ax.tick_params(which="both", direction="in", top=True, right=True, length=2.4, width=0.5, labelsize=7, pad=2)
    ax.tick_params(which="minor", length=1.3)
    return fig, ax


def single_3d(rect=(0.04, 0.10, 0.92, 0.90)):
    fig, axa, axb = new_figure()
    axa.remove(); axb.remove()
    for t in list(fig.texts):
        t.remove()
    return fig, fig.add_axes(rect, projection="3d", computed_zorder=False)


def sk1(out):
    b, d, n0, T, runs = 0.12, 0.06, 20, 25.0, 2000
    rng = np.random.default_rng(20261104)
    grid = np.linspace(0, T, 11)

    def gillespie():
        t, n, ts, ns = 0.0, n0, [0.0], [n0]
        while n > 0:
            t += rng.exponential(1 / ((b + d) * n))
            if t > T:
                break
            n += 1 if rng.random() < b / (b + d) else -1
            ts.append(t); ns.append(n)
        return np.array(ts), np.array(ns)

    samples, shown = np.zeros((runs, len(grid))), []
    for r in range(runs):
        ts, ns = gillespie()
        samples[r] = ns[np.searchsorted(ts, grid, side="right") - 1]
        if r < 12:
            shown.append((ts, ns))
    r = b - d
    mean = n0 * np.exp(r * grid)
    var = n0 * (b + d) / r * np.exp(r * grid) * (np.exp(r * grid) - 1)
    se = np.sqrt(var[1:] / runs)
    z = (samples.mean(axis=0)[1:] - mean[1:]) / se
    print(f"sk1: {runs} runs, mean at T {samples[:, -1].mean():.2f} against exact {mean[-1]:.2f}, largest |z| {np.abs(z).max():.2f}")
    assert np.abs(z).max() < 4
    fig, ax = single_axes()
    for k, (ts, ns) in enumerate(shown):
        c = (ink(RED), ink(GREEN), ink(BLUE))[k] if k < 3 else "#b4b8c0"
        ax.step(ts, ns, where="post", lw=0.8 if k < 3 else 0.45, color=c, zorder=2 if k < 3 else 1)
    ax.plot(grid, mean, "--", color=SLOPE, lw=1.0, zorder=3)
    ax.scatter(grid, samples.mean(axis=0), s=24, c=DEEP, ec="none", zorder=4, clip_on=False)
    ax.set_xlim(0, T); ax.set_ylim(0, 1.05 * max(n.max() for _, n in shown))
    ax.set_xlabel("Time", labelpad=1.5); ax.set_ylabel("Molecules", labelpad=1.5)
    ax.text(0.03, 0.90, f"Dots: mean of {runs:,} runs", transform=ax.transAxes, fontsize=6.5, color=ink(DEEP))
    ax.text(0.03, 0.80, "Dashed: exact mean", transform=ax.transAxes, fontsize=6.5, color=ink(SLOPE))
    save(fig, out / "sk1_programming.svg")


def sk2(out):
    x, y = sp.symbols("x y")
    A, a, bb, c, x0, y0 = (-200, -100, -170, 15), (-1, -1, -6.5, 0.7), (0, 0, 11, 0.6), (-10, -10, -6.5, 0.7), (1, 0, -0.5, -1), (0, 0.5, 1.5, 1)
    V = sum(A[i] * sp.exp(a[i] * (x - x0[i]) ** 2 + bb[i] * (x - x0[i]) * (y - y0[i]) + c[i] * (y - y0[i]) ** 2) for i in range(4))
    f = sp.lambdify((x, y), V, "numpy")
    g = sp.lambdify((x, y), [V.diff(x), V.diff(y)], "numpy")
    h = sp.lambdify((x, y), sp.hessian(V, (x, y)), "numpy")
    minima = [minimize(lambda p: f(*p), s, jac=lambda p: np.array(g(*p), float), tol=1e-10).x for s in ((-0.5, 1.5), (0.6, 0.0), (0.0, 0.5))]
    saddles = [root(lambda p: np.array(g(*p), float), s, jac=lambda p: np.array(h(*p), float), tol=1e-12).x for s in ((-0.8, 0.6), (-0.2, 0.3))]
    Emin, Esad = [f(*p) for p in minima], [f(*p) for p in saddles]
    print("sk2: minima", np.round(Emin, 2), "saddles", np.round(Esad, 2))
    # Published values for this surface, from memory: -146.70, -108.17, -80.77 and -40.66, -72.25
    assert np.allclose(Emin, (-146.70, -108.17, -80.77), atol=0.01) and np.allclose(Esad, (-40.66, -72.25), atol=0.01)
    assert all(np.linalg.eigvalsh(np.array(h(*p), float)).min() > 0 for p in minima)
    assert all(np.linalg.det(np.array(h(*p), float)) < 0 for p in saddles)
    X, Y = np.meshgrid(np.linspace(-1.5, 1.1, 120), np.linspace(-0.4, 2.0, 120))
    Z = np.clip(f(X, Y), None, 40.0)
    fig, ax = single_3d()
    style_3d(ax, 50, -72, r"$x$", r"$y$", "Energy")
    ax.plot_surface(X, Y, Z, cmap=RGB, vmin=-150, vmax=40, linewidth=0, antialiased=True, rcount=120, ccount=120, zorder=1, shade=False, rasterized=True)
    m, s = np.array(minima), np.array(saddles)
    ax.scatter(m[:, 0], m[:, 1], np.array(Emin) + 8, s=26, c=GREEN, ec="none", depthshade=False, zorder=3)
    ax.scatter(s[:, 0], s[:, 1], np.array(Esad) + 8, s=26, c=RED, ec="none", depthshade=False, zorder=3)
    ax.set_xlim(-1.5, 1.1); ax.set_ylim(-0.4, 2.0); ax.set_zlim(-150, 40)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([0, 1, 2]); ax.set_zticks([-150, 0])
    ax.set_box_aspect((1.25, 1, 0.5), zoom=1.22)
    key(fig, [(4, GREEN, 1.2, "Minimum"), (18, RED, 1.2, "Saddle point")])
    save(fig, out / "sk2_software.svg")


def sk3(out):
    k1, km1, k2, E0, S0 = 1.0, 1.0, 1.0, 0.5, 10.0
    Km, Vmax = (km1 + k2) / k1, k2 * E0

    def rhs(_, z):
        S, ES, P = z
        E = E0 - ES
        return [km1 * ES - k1 * E * S, k1 * E * S - (km1 + k2) * ES, k2 * ES]

    t = np.linspace(0, 40, 21)
    sol = solve_ivp(rhs, (0, 40), [S0, 0, 0], t_eval=t, method="LSODA", rtol=1e-10, atol=1e-12)
    S, ES, P = sol.y
    assert np.abs(S + ES + P - S0).max() < 1e-7
    fine = np.linspace(0, 40, 200)
    exact = lambda T: Km * lambertw(S0 / Km * np.exp((S0 - Vmax * T) / Km)).real
    gap = np.abs(S[1:] - exact(t[1:])).max() / S0
    print(f"sk3: mass balance holds, largest |ODE - Lambert W|/S0 = {gap:.3f}, Km={Km}, Vmax={Vmax}")
    assert gap < 0.06
    fig, axa, axb = new_figure()
    nodes = {"S": (-1.7, 0.1, RED, 0.34), "E": (-0.2, 1.2, GREEN, 0.28), "ES": (0.2, -0.75, BLUE, 0.40), "P": (1.8, 0.45, PURPLE, 0.34)}
    rx = {"r1": (-0.55, 0.15), "r2": (0.85, 0.35)}
    tilt, cam = np.radians(38), 9.0
    proj = lambda p: np.array((p[0], p[1] * np.cos(tilt))) * cam / (cam + p[1] * np.sin(tilt))
    links = [("S", "r1"), ("E", "r1"), ("r1", "ES"), ("ES", "r2"), ("r2", "E"), ("r2", "P")]
    pos = {k: proj(v[:2]) for k, v in nodes.items()} | {k: proj(v) for k, v in rx.items()}
    for n, v in nodes.items():
        axa.add_patch(Ellipse(pos[n] - np.array((0, 0.42)), 0.8, 0.28, fc=SHADOW, ec="none", alpha=0.35, zorder=0))
    for u, w in links:
        ru = nodes[u][3] if u in nodes else 0.07
        rw = nodes[w][3] if w in nodes else 0.07
        d = pos[w] - pos[u]; d = d / np.linalg.norm(d)
        axa.annotate("", xy=pos[w] - d * rw, xytext=pos[u] + d * ru, arrowprops=dict(arrowstyle="-|>,head_width=0.14,head_length=0.28", lw=0.9, color=STICK, shrinkA=0, shrinkB=0), zorder=1)
    for k in rx:
        axa.add_patch(plt.Circle(pos[k], 0.07, fc="#3a3d44", ec="none", zorder=2))
    for n, (_, _, c, r) in nodes.items():
        axa.add_patch(plt.Circle(pos[n], r, fc=c, ec="none", zorder=3))
        axa.text(pos[n][0], pos[n][1], n, color="white", fontsize=7, weight="bold", ha="center", va="center", zorder=4)
    axa.set_aspect("equal"); axa.axis("off"); axa.set_xlim(-2.4, 2.5); axa.set_ylim(-1.3, 1.5)
    key(fig, [(3, RED, 1.1, "Substrate"), (18, GREEN, 1.1, "Enzyme"), (30.5, BLUE, 1.1, "Complex"), (43.5, PURPLE, 1.1, "Product")])
    axb.plot(fine, exact(fine), "--", color=SLOPE, lw=0.9, zorder=2)
    axb.scatter(t, S, s=11, c=RED, ec="none", zorder=3, clip_on=False)
    axb.scatter(t, P, s=11, c=PURPLE, ec="none", zorder=3, clip_on=False)
    axb.scatter(t, ES, s=11, c=BLUE, ec="none", zorder=3, clip_on=False)
    axb.set_xlim(0, 40); axb.set_ylim(0, 10.5); axb.set_xticks([0, 20, 40])
    axb.set_xlabel("Time", labelpad=1.5); axb.set_ylabel("Concentration", labelpad=1.5)
    save(fig, out / "sk3_methods.svg")


def run_stats():
    text = (HERE / "pytie_7agnr_output.txt").read_text(encoding="utf-8")
    found = re.findall(r"N\.V of (.+?) =\s+\[(.*?)\]", text)
    names, vals = [], []
    for name, body in found:
        names.append(name.strip()); vals.append([float(v) for v in body.split(",")])
    return names, np.array(vals)


def sk4(out):
    names, vals = run_stats()
    print(f"sk4: PyTIE printed {len(names)} descriptors, {vals.shape[1]} sizes")
    assert len(names) == 25 and (vals > 0).all()
    # Cross-check PyTIE against rc6_compute.py on three descriptors, by position in the PyTIE output
    from rc6_compute import ribbon, indices
    x = ribbon(7, 7)
    mine, _ = indices(x)
    assert abs(vals[0, 5] - mine["M1"][0]) < 1e-6
    assert abs(vals[6, 5] - mine["R"][0]) < 1e-6
    assert abs(vals[10, 5] - mine["ABC"][0]) < 1e-6
    print(f"sk4: PyTIE agrees with rc6 at N=94: M1 {vals[0, 5]:.3f}, R {vals[6, 5]:.3f}, ABC {vals[10, 5]:.3f}")
    atoms = 24 + 14 * np.arange(vals.shape[1])
    order = np.argsort(vals[:, -1])
    fig, ax = single_3d((0.02, 0.10, 0.96, 0.90))
    style_3d(ax, 22, -60, "Descriptor rank", r"Atoms, $N$", r"log$_{10}$ value")
    X, Y = np.meshgrid(np.arange(1, 26), atoms)
    Z = np.log10(vals[order].T)
    fl = np.floor(Z.min()) - 0.2
    for xx, yy, zz in zip(X.ravel(), Y.ravel(), Z.ravel()):
        ax.plot([xx, xx], [yy, yy], [fl, zz], color="#c3c7cf", lw=0.3, zorder=1)
    ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), s=7, c=GREEN, ec="none", depthshade=False, zorder=2)
    ax.scatter(X[-1], Y[-1], Z[-1], s=11, c=RED, ec="none", depthshade=False, zorder=3)
    ax.set_xlim(0, 26); ax.set_ylim(atoms[0] - 5, atoms[-1] + 5); ax.set_zlim(fl, Z.max() + 0.2)
    ax.set_xticks([1, 25]); ax.set_yticks([atoms[0], atoms[-1]]); ax.set_zticks([0, 2, 4, 6][:int(Z.max() // 2) + 1])
    ax.set_box_aspect((1.5, 1, 0.75), zoom=1.1)
    key(fig, [(4, GREEN, 1.0, "PyTIE output"), (23, RED, 1.0, f"Largest ribbon, {atoms[-1]} atoms")])
    save(fig, out / "sk4_built.svg")


def sk5(out):
    rng = np.random.default_rng(20261104)

    def moons(n):
        t = rng.uniform(0, np.pi, n)
        c = rng.integers(0, 2, n)
        p = np.column_stack((np.cos(t), np.sin(t)))
        p = np.where(c[:, None] == 0, p, np.column_stack((1 - np.cos(t), 0.5 - np.sin(t))))
        return p + rng.normal(0, 0.16, (n, 2)), c

    Xtr, ytr = moons(160)
    Xte, yte = moons(1000)
    H = 16
    w = [rng.normal(0, 0.7, (2, H)), np.zeros(H), rng.normal(0, 0.7, (H, 1)), np.zeros(1)]

    def forward(w, X):
        h = np.tanh(X @ w[0] + w[1])
        return h, 1 / (1 + np.exp(-(h @ w[2] + w[3])[:, 0]))

    def loss(w, X, y):
        p = np.clip(forward(w, X)[1], 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def grads(w, X, y):
        h, p = forward(w, X)
        e = (p - y)[:, None] / len(y)
        dh = e @ w[2].T
        s = 1 - h ** 2
        return [X.T @ (dh * s), (dh * s).sum(axis=0), h.T @ e, e.sum(axis=0)]

    g = grads(w, Xtr, ytr)
    u = [a.copy() for a in w]
    eps = 1e-6
    u[0][0, 0] += eps; hi = loss(u, Xtr, ytr)
    u[0][0, 0] -= 2 * eps; lo = loss(u, Xtr, ytr)
    assert abs((hi - lo) / (2 * eps) - g[0][0, 0]) < 1e-7, "gradient check"
    m1, m2 = [np.zeros_like(a) for a in w], [np.zeros_like(a) for a in w]
    for it in range(1, 3001):
        g = grads(w, Xtr, ytr)
        for i in range(4):
            m1[i] = 0.9 * m1[i] + 0.1 * g[i]
            m2[i] = 0.999 * m2[i] + 0.001 * g[i] ** 2
            w[i] = w[i] - 0.01 * (m1[i] / (1 - 0.9 ** it)) / (np.sqrt(m2[i] / (1 - 0.999 ** it)) + 1e-8)
    tr = ((forward(w, Xtr)[1] > 0.5) == ytr).mean()
    te = ((forward(w, Xte)[1] > 0.5) == yte).mean()
    print(f"sk5: gradient check passed, final loss {loss(w, Xtr, ytr):.3f}, train accuracy {tr:.3f}, held-out accuracy on 1000 points {te:.3f}")
    assert te > 0.9
    xg, yg = np.meshgrid(np.linspace(-1.5, 2.5, 110), np.linspace(-1.1, 1.6, 110))
    Pg = forward(w, np.column_stack((xg.ravel(), yg.ravel())))[1].reshape(xg.shape)
    fig, ax = single_3d()
    style_3d(ax, 30, -62, r"$x_1$", r"$x_2$", "Probability")
    ax.plot_surface(xg, yg, Pg, cmap=LinearSegmentedColormap.from_list("rb", [BLUE, "#e8e8ec", RED]), vmin=0, vmax=1, linewidth=0, antialiased=True, rcount=110, ccount=110, shade=False, zorder=1, rasterized=True)
    for c, col in ((0, BLUE), (1, RED)):
        m = ytr == c
        ax.scatter(Xtr[m, 0], Xtr[m, 1], c * 1.0 + (0.03 if c else -0.03), s=7, c=col, ec="none", depthshade=False, zorder=3)
    ax.set_xlim(-1.5, 2.5); ax.set_ylim(-1.1, 1.6); ax.set_zlim(-0.05, 1.05)
    ax.set_xticks([-1, 2]); ax.set_yticks([-1, 1]); ax.set_zticks([0, 1])
    ax.set_box_aspect((1.3, 1, 0.65), zoom=1.15)
    key(fig, [(4, BLUE, 1.0, "Class 0"), (15, RED, 1.0, "Class 1")])
    fig.text(0.97, 0.052, f"Held-out accuracy {te * 100:.1f}%", fontsize=6.5, ha="right", va="center")
    save(fig, out / "sk5_certification.svg")


def sk6(out):
    vowels = [chr(c) for c in (0x0B85, 0x0B86, 0x0B87, 0x0B88, 0x0B89, 0x0B8A, 0x0B8E, 0x0B8F, 0x0B90, 0x0B92, 0x0B93, 0x0B94)]
    signs = ["", *[chr(c) for c in (0x0BBE, 0x0BBF, 0x0BC0, 0x0BC1, 0x0BC2, 0x0BC6, 0x0BC7, 0x0BC8, 0x0BCA, 0x0BCB, 0x0BCC)]]
    bases = [chr(c) for c in (0x0B95, 0x0B99, 0x0B9A, 0x0B9E, 0x0B9F, 0x0BA3, 0x0BA4, 0x0BA8, 0x0BAA, 0x0BAE, 0x0BAF, 0x0BB0, 0x0BB2, 0x0BB5, 0x0BB4, 0x0BB3, 0x0BB1, 0x0BA9)]
    pulli, aytham = chr(0x0BCD), chr(0x0B83)
    english = [chr(c) for c in range(65, 91)]
    combined = [b + s for b in bases for s in signs]
    total = len(vowels) + len(bases) + len(combined) + 1
    print(f"sk6: Tamil {len(vowels)} vowels + {len(bases)} consonants + {len(combined)} combined + 1 aytham = {total}, English {len(english)}")
    assert total == 247 and len(set(combined)) == 216 and len(english) == 26
    font = "font-family=\"'Nirmala UI','Noto Sans Tamil',Latha,sans-serif\""
    cw, ch, x0, y0 = 38.6, 11.7, 13.0, 27.0
    t = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200" role="img" aria-label="Tamil and English alphabets"><rect width="400" height="200" fill="#ffffff"/>']
    t.append(f'<text x="8" y="12" font-family="Arial,sans-serif" font-size="8.5" fill="#222222"><tspan font-weight="700">Tamil</tspan> {total} letters: <tspan fill="{RED}">{len(vowels)} vowels</tspan>, <tspan fill="{BLUE}">{len(bases)} consonants</tspan>, {len(combined)} combined, <tspan fill="{GREEN}">1 āytham</tspan>. First 9 consonants shown.</text>')
    t.append(f'<g {font} font-size="10.2" text-anchor="middle">')
    t.append(f'<text x="{x0:.1f}" y="{y0:.1f}" fill="{GREEN}">{aytham}</text>')
    for i, b in enumerate(bases[:9]):
        t.append(f'<text x="{x0 + (i + 1) * cw:.1f}" y="{y0:.1f}" fill="{BLUE}">{b}{pulli}</text>')
    for j, v in enumerate(vowels):
        y = y0 + (j + 1) * ch
        t.append(f'<text x="{x0:.1f}" y="{y:.1f}" fill="{RED}">{v}</text>')
        for i, b in enumerate(bases[:9]):
            t.append(f'<text x="{x0 + (i + 1) * cw:.1f}" y="{y:.1f}" fill="#2b2d33">{b}{signs[j]}</text>')
    t.append('</g>')
    t.append(f'<text x="8" y="181" font-family="Arial,sans-serif" font-size="8.5" fill="#222222"><tspan font-weight="700">English</tspan> {len(english)} letters</text>')
    t.append('<g font-family="Arial,sans-serif" font-size="9.5" text-anchor="middle" fill="#2b2d33">')
    for i, c in enumerate(english):
        t.append(f'<text x="{11 + i * 15.1:.1f}" y="194">{c}</text>')
    t.append('</g></svg>')
    (out / "sk6_languages.svg").write_text("".join(t), encoding="utf-8")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    for fn in (sk1, sk2, sk3, sk4, sk5, sk6):
        fn(out)
