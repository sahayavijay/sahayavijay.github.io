"""Homepage figure rc1: exact graph calculations on a hydrogen-bonded boric acid sheet.

Graph: vertices are B, O and H atoms. Edges are B-O and O-H covalent bonds and O-H...O hydrogen bonds.
Geometry (idealised, planar, used only for drawing): B-O 1.36, O-H 0.97, O...O 2.72, B...B 4.08 angstrom.
Distances are topological: every edge has length 1, so geometry does not enter W.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Ellipse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components

BO, OH, OO = 1.36, 0.97, 2.72
BB = BO + OO
RADIUS = {"B": 0.66, "O": 0.50, "H": 0.30}
# Dark element colours, and the sphere colour for panel b
ATOM = {"B": "#12703a", "O": "#b01c1c", "H": "#1b45a8"}
BLUE = "#0b4f8a"
SHADOW = "#c9ccd2"


def shade(colour, f):
    """Mix a hex colour towards white (f > 0) or black (f < 0)."""
    c = np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], float)
    c = c + (255 - c) * f if f > 0 else c * (1 + f)
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in c)


def add_sphere_gradients(path):
    """Replace flat sphere fills in the SVG with radial gradients, and blur the shadows."""
    svg = Path(path).read_text(encoding="utf-8")
    defs = '<filter id="soft" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="1.8"/></filter>'
    for i, c in enumerate((*ATOM.values(), BLUE)):
        if f"fill: {c}" not in svg:
            continue
        svg = svg.replace(f"fill: {c}", f"fill: url(#s{i})")
        defs += (f'<radialGradient id="s{i}" cx="0.42" cy="0.38" r="0.62" fx="0.32" fy="0.26">'
                 f'<stop offset="0" stop-color="{shade(c, 0.62)}"/><stop offset="0.36" stop-color="{c}"/>'
                 f'<stop offset="1" stop-color="{shade(c, -0.65)}"/></radialGradient>')
    assert f"fill: {SHADOW}" in svg
    svg = svg.replace(f"fill: {SHADOW}", f"fill: {SHADOW}; filter: url(#soft)")
    svg = svg.replace("<defs>", "<defs>" + defs, 1)
    Path(path).write_text(svg, encoding="utf-8")


def flake(n):
    """Honeycomb flake of molecules made of all hexagons within n-1 steps of the centre."""
    a1, a2 = np.array([np.sqrt(3) * BB, 0.0]), np.array([np.sqrt(3) / 2 * BB, 1.5 * BB])
    sites = {}
    for q in range(-n + 1, n):
        for r in range(-n + 1, n):
            if max(abs(q), abs(r), abs(q + r)) > n - 1:
                continue
            c = q * a1 + r * a2
            for k in range(6):
                t = np.radians(30 + 60 * k)
                p = c + BB * np.array([np.cos(t), np.sin(t)])
                # sublattice: O atoms point away from the three neighbours
                sites[(round(p[0], 3), round(p[1], 3))] = 90 if k % 2 == 0 else 30
    return sites


def build(n):
    xy, kind, edges, hbond = [], [], [], []
    ox_lookup = []
    for (x, y), t0 in flake(n).items():
        b = len(xy)
        xy.append((x, y)); kind.append("B")
        for j in range(3):
            t = np.radians(t0 + 120 * j)
            o = np.array([x, y]) + BO * np.array([np.cos(t), np.sin(t)])
            d = np.array([np.cos(t - np.pi / 3), np.sin(t - np.pi / 3)])  # donor direction
            io = len(xy); xy.append(tuple(o)); kind.append("O")
            ih = len(xy); xy.append(tuple(o + OH * d)); kind.append("H")
            edges += [(b, io), (io, ih)]
            ox_lookup.append((io, ih, o + OO * d))
    xy = np.array(xy)
    oxy = np.array([e[0] for e in ox_lookup])
    for io, ih, target in ox_lookup:
        dist = np.linalg.norm(xy[oxy] - target, axis=1)
        m = dist.argmin()
        if dist[m] < 0.01:
            hbond.append((ih, oxy[m]))
    return xy, np.array(kind), edges, hbond


def distances(n):
    xy, kind, edges, hbond = build(n)
    N = len(xy)
    e = np.array(edges + hbond)
    A = csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(N, N))
    D = shortest_path(A, directed=False, unweighted=True)
    assert connected_components(A, directed=False)[0] == 1
    return xy, kind, edges, hbond, D


def floyd_warshall(N, e):
    D = np.full((N, N), np.inf); np.fill_diagonal(D, 0)
    for u, v in e:
        D[u, v] = D[v, u] = 1
    for k in range(N):
        D = np.minimum(D, D[:, k, None] + D[k, None, :])
    return D


def check():
    # Hand count for one B(OH)3: B-O 3, B-H 6, O-O 6, O-H 3+18, H-H 12 = 48
    xy, kind, edges, hbond = build(1)
    assert len(edges) == 36
    e7 = [(0, 1), (1, 2), (0, 3), (3, 4), (0, 5), (5, 6)]
    assert floyd_warshall(7, e7).sum() / 2 == 48
    for n in (1, 2, 3):
        xy, kind, edges, hbond, D = distances(n)
        assert len(xy) == 42 * n ** 2, "atoms"
        assert len(edges) + len(hbond) == 54 * n ** 2 - 6 * n, "edges"
        assert len(hbond) == 2 * (9 * n ** 2 - 3 * n), "hydrogen bonds"
        assert np.array_equal(D, floyd_warshall(len(xy), edges + hbond)), "BFS vs Floyd-Warshall"
    print("checks passed: hand count W=48, N=42n^2, E=54n^2-6n, BFS = Floyd-Warshall for n=1,2,3")


def main(outdir):
    check()
    ns = np.arange(1, 13)
    Ns, Ws = [], []
    for n in ns:
        *_, D = distances(int(n))
        Ns.append(D.shape[0]); Ws.append(int(D.sum() / 2))
    Ns, Ws = np.array(Ns), np.array(Ws)
    slopes = np.diff(np.log(Ws)) / np.diff(np.log(Ns))
    print(" n     N          W  local slope")
    for i, n in enumerate(ns):
        print(f"{n:2d} {Ns[i]:5d} {Ws[i]:10d}  {slopes[i-1]:.3f}" if i else f"{n:2d} {Ns[i]:5d} {Ws[i]:10d}")

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "mathtext.default": "it", "mathtext.fontset": "custom", "mathtext.it": "Arial:italic", "mathtext.rm": "Arial", "axes.linewidth": 0.7, "svg.fonttype": "path"})
    fig = plt.figure(figsize=(4.0, 2.0))
    axa = fig.add_axes([0.0, 0.105, 0.56, 0.895])
    axb = fig.add_axes([0.665, 0.185, 0.315, 0.73])

    xy, kind, edges, hbond, D = distances(2)
    print(f"panel a: n=2, N={len(xy)}, W={int(D.sum()/2)}, H bonds={len(hbond)}")
    # Rotated, tilted perspective view of the planar sheet, for display only
    tilt, cam, turn, drop = np.radians(44), 60.0, np.radians(14), 1.5
    R = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    r = xy @ R.T
    depth = r[:, 1] * np.sin(tilt)
    scale = cam / (cam + depth)
    pxy = np.column_stack((r[:, 0] * scale, r[:, 1] * np.cos(tilt) * scale))
    for i in np.flatnonzero(kind == "B"):
        rad = 2.1 * scale[i]
        axa.add_patch(Ellipse((pxy[i, 0], pxy[i, 1] - drop * scale[i]), 2 * rad, 2 * rad * np.cos(tilt), fc=SHADOW, ec="none", alpha=0.28, zorder=0))
    for pairs, colour, lw, ls in ((edges, "#6b7078", 1.3, "solid"), (hbond, "#6b7078", 0.7, (0, (1, 1.2)))):
        e = np.array(pairs)
        axa.add_collection(LineCollection(pxy[e], colors=colour, linestyles=ls, zorder=1,
                                          linewidths=lw * scale[e].mean(axis=1)))
    for z, i in enumerate(np.argsort(-depth)):
        axa.add_patch(Circle(pxy[i], RADIUS[kind[i]] * scale[i], fc=ATOM[kind[i]], ec="none", zorder=2 + z))
    axa.set_aspect("equal"); axa.axis("off")
    axa.set_xlim(pxy[:, 0].min() - 0.8, pxy[:, 0].max() + 0.8)
    axa.set_ylim(pxy[:, 1].min() - drop - 1.6, pxy[:, 1].max() + 0.6)
    fig.text(0.01, 0.985, "a", weight="bold", fontsize=9, va="top")
    axk = fig.add_axes([0, 0, 0.56, 0.105]); axk.set_xlim(0, 56); axk.set_ylim(0, 5.25)
    axk.set_aspect("equal"); axk.axis("off")
    for x, k, rad, label in ((6, "B", 1.5, "Boron"), (24, "O", 1.15, "Oxygen"), (41, "H", 0.75, "Hydrogen")):
        axk.add_patch(Circle((x, 2.6), rad, fc=ATOM[k], ec="none"))
        axk.text(x + 2.4, 2.6, label, fontsize=6.5, va="center")

    axb.loglog(Ns, Ns ** 2.5 * Ws[-1] / Ns[-1] ** 2.5, "--", color="#b4561a", lw=0.9, zorder=2)
    axb.scatter(Ns, Ws, s=20, c=BLUE, ec="none", zorder=3, clip_on=False)
    axb.text(0.50, 0.24, "slope 5/2", transform=axb.transAxes, color="#b4561a", fontsize=7)
    axb.set_xlim(25, 1.0e4); axb.set_ylim(2e3, 3e9)
    axb.set_xlabel(r"Atoms, $N$", labelpad=1.5)
    axb.set_ylabel(r"Wiener index, $W$", labelpad=1.5)
    axb.tick_params(which="both", direction="in", top=True, right=True, length=2.4, width=0.5, labelsize=7, pad=2)
    axb.tick_params(which="minor", length=1.3)
    axb.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=12))
    axb.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    fig.text(0.545, 0.985, "b", weight="bold", fontsize=9, va="top")
    out = Path(outdir)
    fig.savefig(out / "rc1_graph.svg", facecolor="white")
    add_sphere_gradients(out / "rc1_graph.svg")


if __name__ == "__main__":
    main(sys.argv[1])
