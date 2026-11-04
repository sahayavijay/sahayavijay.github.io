"""Homepage figure rc6: degree-based descriptors of armchair graphene nanoribbons.

Graph: carbon atoms are vertices and C-C bonds are edges, with hydrogen suppressed.
Indices: first Zagreb M1 = sum(du + dv), Randic R = sum 1/sqrt(du dv), ABC = sum sqrt((du + dv - 2)/(du dv)).
Entropy: S = ln(I) - (1/I) sum f ln f, with f the edge contribution to index I.
Stand-alone script. It computes the same kinds of quantity as PyTIE, but it does not run PyTIE.
"""
import sys
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
from figure_style import *

CC = 1.42
FORMS = {"M1": lambda a, b: a + b, "R": lambda a, b: (a * b) ** -0.5, "ABC": lambda a, b: ((a + b - 2) / (a * b)) ** 0.5}


def graph(xy):
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    i, j = np.nonzero(np.triu((d > 0.1) & (d < 1.1 * CC)))
    deg = np.bincount(np.r_[i, j], minlength=len(xy))
    return i, j, deg


def ribbon(width, cells):
    """Armchair ribbon with a given number of dimer lines, with dangling atoms removed."""
    pts = []
    for row in range(width):
        y = row * CC * np.sqrt(3) / 2
        shift = 0.0 if row % 2 == 0 else 1.5 * CC
        for c in range(cells + 1):
            for x0 in (0.0, CC):
                pts.append((shift + x0 + 3 * CC * c, y))
    xy = np.array(pts)
    xy = xy[xy[:, 0] <= 3 * CC * cells + 1e-6]
    while True:
        *_, deg = graph(xy)
        if (deg >= 2).all():
            return xy
        xy = xy[deg >= 2]


def indices(xy):
    i, j, deg = graph(xy)
    out = {}
    for name, g in FORMS.items():
        f = g(deg[i].astype(float), deg[j].astype(float))
        I = f.sum()
        out[name] = (I, np.log(I) - (f * np.log(f)).sum() / I)
    return out, len(i)


def check():
    # Hand values for coronene: edges (2,2) 6, (2,3) 12, (3,3) 12
    t = np.radians(30 + 60 * np.arange(6))
    centres = np.vstack(([0, 0], CC * np.sqrt(3) * np.column_stack((np.cos(t), np.sin(t)))))
    t2 = np.radians(60 * np.arange(6))
    pts = np.round(np.vstack([c + CC * np.column_stack((np.cos(t2), np.sin(t2))) for c in centres]), 4)
    xy = np.unique(pts, axis=0)
    assert len(xy) == 24
    out, E = indices(xy)
    assert E == 30
    assert abs(out["M1"][0] - 156) < 1e-9
    assert abs(out["R"][0] - (3 + 12 / np.sqrt(6) + 4)) < 1e-9
    assert abs(out["ABC"][0] - (6 * np.sqrt(.5) + 12 * np.sqrt(.5) + 12 * np.sqrt(4 / 9))) < 1e-9
    print(f"check passed: coronene hand values, 24 atoms, 30 bonds, M1=156, R={out['R'][0]:.3f}, ABC={out['ABC'][0]:.3f}")


def main(outdir):
    check()
    fig, axa, axb = new_figure()
    xy = ribbon(7, 7)
    i, j, deg = graph(xy)
    print(f"panel a: N={len(xy)}, E={len(i)}, degree 2 atoms {(deg == 2).sum()}, degree 3 atoms {(deg == 3).sum()}, max degree {deg.max()}")
    assert deg.max() == 3 and deg.min() == 2
    xy = xy - xy.mean(axis=0)
    t = np.radians(12)
    r = xy @ np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    tilt, cam = np.radians(46), 62.0
    depth = r[:, 1] * np.sin(tilt)
    scale = cam / (cam + depth)
    p = np.column_stack((r[:, 0] * scale, r[:, 1] * np.cos(tilt) * scale))
    for n in range(len(p)):
        axa.add_patch(Ellipse((p[n, 0], p[n, 1] - 1.5 * scale[n]), 1.8, 1.1, fc=SHADOW, ec="none", alpha=0.20, zorder=0))
    axa.add_collection(LineCollection(np.stack((p[i], p[j]), axis=1), colors=STICK, linewidths=1.3 * (scale[i] + scale[j]) / 2, zorder=1))
    for z, n in enumerate(np.argsort(-depth)):
        axa.add_patch(plt.Circle(p[n], (0.40 if deg[n] == 3 else 0.33) * scale[n], fc=GREEN if deg[n] == 3 else BLUE, ec="none", zorder=2 + z))
    axa.set_aspect("equal"); axa.axis("off")
    axa.set_xlim(p[:, 0].min() - 0.6, p[:, 0].max() + 0.6); axa.set_ylim(p[:, 1].min() - 2.8, p[:, 1].max() + 0.9)
    key(fig, [(5, GREEN, 1.3, "Degree 3 carbon"), (30, BLUE, 1.05, "Degree 2 carbon")])

    cells = np.array([2, 3, 4, 6, 8, 11, 15, 20, 28, 40, 56, 80])
    N, data = [], {k: [] for k in FORMS}
    ent = None
    for c in cells:
        x = ribbon(7, int(c))
        out, E = indices(x)
        N.append(len(x))
        for name in FORMS:
            data[name].append(out[name][0])
        ent = out, len(x)
    N = np.array(N)
    print("N, M1, R, ABC")
    for k in range(len(N)):
        print(N[k], *(round(data[n][k], 3) for n in FORMS))
    per = {n: (data[n][-1] - data[n][-2]) / (N[-1] - N[-2]) for n in FORMS}
    print("increase per atom:", {n: round(v, 4) for n, v in per.items()})
    print(f"entropy at N={ent[1]}:", {n: round(ent[0][n][1], 4) for n in FORMS})
    for name, c, label, xt, yt in (("M1", RED, "$M_1$", 110, 2.6e3), ("ABC", GREEN, "ABC", 130, 3.2e2), ("R", BLUE, "$R$", 480, 1.25e2)):
        axb.loglog(N, N * per[name] + (data[name][-1] - N[-1] * per[name]), "--", color=SLOPE, lw=0.7, zorder=2)
        axb.scatter(N, data[name], s=11, c=c, ec="none", zorder=3, clip_on=False)
        axb.text(xt, yt, label, color=ink(c), fontsize=6.5, ha="center", va="center")
    axb.text(0.05, 0.92, f"Entropy of $R$ at $N$ = {ent[1]:,}: {ent[0]['R'][1]:.2f}", transform=axb.transAxes, fontsize=5.8, va="center")
    axb.set_xlim(20, 2000); axb.set_ylim(10, 2e4)
    axb.set_xlabel(r"Carbon atoms, $N$", labelpad=1.5)
    axb.set_ylabel("Descriptor value", labelpad=1.5)
    save(fig, Path(outdir) / "rc6_compute.svg")


if __name__ == "__main__":
    main(sys.argv[1])
