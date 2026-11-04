"""Small homepage Skills figures. Each one is simple, and each one comes from something that really runs.

sk1 Programming: a short Python function, executed here so its printed result is true.
sk2 Scientific software: a rate law solved symbolically with SymPy, then typeset.
sk3 Research methods: A -> B solved as an ODE and checked against the exact solution.
sk4 Software I built: the naphthalene graph with its vertex degrees and first Zagreb index.
sk5 Certifications: a logistic regression trained with NumPy on generated data.
sk6 Languages: a greeting in Tamil and English.
"""
import contextlib
import io
import sys
import sympy as sp
from scipy.integrate import solve_ivp
from figure_style import *
from rc6_compute import graph, indices, CC

WIDTH, HEIGHT = 1.8, 1.2  # inches, shown at 172 px wide


def small(rect=(0, 0, 1, 1)):
    new_figure()
    plt.close("all")
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    return fig, fig.add_axes(rect)


def sk1(out):
    code = "def grow(x, k, dt):\n    return x + k*x*dt\nprint(grow(1.0, 0.5, 0.1))"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code)
    result = buf.getvalue().strip()
    print("sk1: the code prints", result)
    assert result == "1.05"
    kw, name, num, plain, grey = "#b01c1c", "#1b45a8", "#12703a", "#2b2d33", "#6b7078"
    lines = [
        [("def ", kw), ("grow", name), ("(x, k, dt):", plain)],
        [("    return ", kw), ("x + k*x*dt", plain)],
        [("print", name), ("(grow(", plain), ("1.0", num), (", ", plain), ("0.5", num), (", ", plain), ("0.1", num), ("))", plain)],
        [(result, grey)],
    ]
    t = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 172 115" role="img" aria-label="A short Python function and its result">',
         '<rect width="172" height="115" fill="#ffffff"/>',
         '<g font-family="Consolas,\'DejaVu Sans Mono\',Menlo,monospace" font-size="10.6">']
    for i, parts in enumerate(lines):
        y = 34 + i * 18 + (5 if i == 3 else 0)
        spans = "".join(f'<tspan fill="{c}">{s}</tspan>' for s, c in parts)
        t.append(f'<text x="7" y="{y}" xml:space="preserve">{spans}</text>')
    t.append('</g><line x1="7" y1="77" x2="165" y2="77" stroke="#d5d8de" stroke-width="0.8"/></svg>')
    (out / "sk1_programming.svg").write_text("".join(t), encoding="utf-8")


def sk2(out):
    t, k, x0 = sp.symbols("t k x_0", positive=True)
    x = sp.Function("x")
    sol = sp.dsolve(x(t).diff(t) + k * x(t), x(t), ics={x(0): x0})
    print("sk2: SymPy gives", sol)
    assert sp.simplify(sol.rhs - x0 * sp.exp(-k * t)) == 0
    fig, ax = small()
    ax.axis("off")
    ax.text(0.5, 0.70, r"$\frac{dx}{dt} = -kx$", fontsize=12, ha="center", va="center", color="#2b2d33")
    ax.annotate("", xy=(0.5, 0.40), xytext=(0.5, 0.54), arrowprops=dict(arrowstyle="-|>,head_width=0.14,head_length=0.28", lw=0.8, color=STICK))
    ax.text(0.5, 0.24, r"$x(t) = x_0\,e^{-kt}$", fontsize=12, ha="center", va="center", color=ink(BLUE))
    save(fig, out / "sk2_software.svg")


def sk3(out):
    k = 0.5
    t = np.linspace(0, 10, 200)
    sol = solve_ivp(lambda _, z: [-k * z[0], k * z[0]], (0, 10), [1, 0], t_eval=t, method="DOP853", rtol=1e-10, atol=1e-12)
    A, B = sol.y
    err = max(np.abs(A - np.exp(-k * t)).max(), np.abs(B - 1 + np.exp(-k * t)).max())
    print(f"sk3: largest difference from the exact solution {err:.1e}")
    assert err < 1e-8
    fig, ax = small((0.13, 0.20, 0.79, 0.70))
    ax.plot(t, A, color=ink(RED), lw=1.2)
    ax.plot(t, B, color=ink(BLUE), lw=1.2)
    ax.scatter([0], [A[0]], s=62, c=RED, ec="none", zorder=3, clip_on=False)
    ax.scatter([10], [B[-1]], s=62, c=BLUE, ec="none", zorder=3, clip_on=False)
    ax.text(0, A[0], "A", color="white", fontsize=6.5, weight="bold", ha="center", va="center", zorder=4)
    ax.text(10, B[-1], "B", color="white", fontsize=6.5, weight="bold", ha="center", va="center", zorder=4)
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.0); ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Time", fontsize=7, labelpad=2)
    ax.set_ylabel("Amount", fontsize=7, labelpad=2)
    save(fig, out / "sk3_methods.svg")


def sk4(out):
    centres = np.array([[-np.sqrt(3) / 2 * CC, 0], [np.sqrt(3) / 2 * CC, 0]])
    a = np.radians(30 + 60 * np.arange(6))
    xy = np.unique(np.round(np.vstack([c + CC * np.column_stack((np.cos(a), np.sin(a))) for c in centres]), 4), axis=0)
    values, E = indices(xy)
    i, j, deg = graph(xy)
    m1 = values["M1"][0]
    print(f"sk4: naphthalene has {len(xy)} atoms, {E} bonds, M1 = {m1:.0f}")
    assert len(xy) == 10 and E == 11 and abs(m1 - 50) < 1e-9
    fig, ax = small((0.08, 0.20, 0.84, 0.76))
    for u, v in zip(i, j):
        ax.plot(xy[[u, v], 0], xy[[u, v], 1], color=STICK, lw=1.1, zorder=1)
    for p, d in zip(xy, deg):
        ax.add_patch(plt.Circle(p, 0.36, fc=GREEN if d == 3 else BLUE, ec="none", zorder=2))
        ax.text(*p, str(d), color="white", fontsize=6.5, weight="bold", ha="center", va="center", zorder=3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(xy[:, 0].min() - 0.5, xy[:, 0].max() + 0.5); ax.set_ylim(xy[:, 1].min() - 0.5, xy[:, 1].max() + 0.5)
    fig.text(0.5, 0.09, f"Zagreb index = {m1:.0f}", fontsize=7, ha="center", va="center", color="#2b2d33")
    save(fig, out / "sk4_built.svg")


def sk5(out):
    rng = np.random.default_rng(20261104)

    def data(n):
        y = np.r_[np.zeros(n), np.ones(n)]
        X = rng.normal(0, 0.55, (2 * n, 2)) + np.where(y[:, None] == 0, [-1.0, -0.7], [1.0, 0.7])
        return X, y

    Xtr, ytr = data(14)
    Xte, yte = data(500)
    w, b = np.zeros(2), 0.0
    for _ in range(2000):
        p = 1 / (1 + np.exp(-(Xtr @ w + b)))
        w -= 0.1 * Xtr.T @ (p - ytr) / len(ytr)
        b -= 0.1 * (p - ytr).mean()
    acc = (((Xte @ w + b) > 0) == yte).mean()
    print(f"sk5: accuracy on 1000 unseen points {acc:.3f}")
    assert acc > 0.9
    fig, ax = small((0.05, 0.06, 0.90, 0.88))
    xs = np.array([-3, 3])
    ax.plot(xs, -(w[0] * xs + b) / w[1], "--", color=SLOPE, lw=1.0, zorder=1)
    ax.scatter(*Xtr[ytr == 0].T, s=24, c=BLUE, ec="none", zorder=2)
    ax.scatter(*Xtr[ytr == 1].T, s=24, c=RED, ec="none", zorder=2)
    ax.set_xlim(-3, 3); ax.set_ylim(-2.5, 2.5); ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_linewidth(0.5); sp_.set_color("#b8bdc6")
    save(fig, out / "sk5_certification.svg")


def sk6(out):
    tamil = "".join(chr(c) for c in (0x0BB5, 0x0BA3, 0x0B95, 0x0BCD, 0x0B95, 0x0BAE, 0x0BCD))
    print("sk6: greeting", tamil.encode("unicode_escape").decode())
    (out / "sk6_languages.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 172 115" role="img" aria-label="Hello in Tamil and in English">'
        '<rect width="172" height="115" fill="#ffffff"/>'
        f'<text x="86" y="50" text-anchor="middle" font-family="\'Nirmala UI\',\'Noto Sans Tamil\',Latha,sans-serif" font-size="25" fill="{RED}">{tamil}</text>'
        f'<text x="86" y="89" text-anchor="middle" font-family="Georgia,\'Times New Roman\',serif" font-size="25" fill="{BLUE}">Hello</text>'
        '</svg>', encoding="utf-8")


if __name__ == "__main__":
    target = Path(sys.argv[1])
    for fn in (sk1, sk2, sk3, sk4, sk5, sk6):
        fn(target)
