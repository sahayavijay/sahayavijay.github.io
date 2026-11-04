"""Shared style for the homepage research figures (rc2 to rc6). rc1_graph.py set this look."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

GREEN, RED, BLUE, DEEP, PURPLE = "#12703a", "#b01c1c", "#1b45a8", "#0b4f8a", "#5b2a86"
SLOPE, STICK, SHADOW = "#b4561a", "#6b7078", "#c9ccd2"
SPHERES = (GREEN, RED, BLUE, DEEP, PURPLE)


def ink(colour):
    """A near-identical colour for text and lines, so sphere shading leaves it alone."""
    return "#%06x" % (int(colour[1:], 16) + 1)


def shade(colour, f):
    """Mix a hex colour towards white (f > 0) or black (f < 0)."""
    c = np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], float)
    c = c + (255 - c) * f if f > 0 else c * (1 + f)
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in c)


def new_figure(panel_a_3d=False):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.linewidth": 0.7, "svg.fonttype": "path",
                         "mathtext.default": "it", "mathtext.fontset": "custom",
                         "mathtext.it": "Arial:italic", "mathtext.rm": "Arial"})
    fig = plt.figure(figsize=(4.0, 2.0))
    if panel_a_3d:
        axa = fig.add_axes([-0.01, 0.125, 0.55, 0.875], projection="3d", computed_zorder=False)
    else:
        axa = fig.add_axes([0.0, 0.105, 0.56, 0.895])
    axb = fig.add_axes([0.665, 0.185, 0.315, 0.73])
    fig.text(0.01, 0.985, "a", weight="bold", fontsize=9, va="top")
    fig.text(0.545, 0.985, "b", weight="bold", fontsize=9, va="top")
    axb.tick_params(which="both", direction="in", top=True, right=True, length=2.4, width=0.5, labelsize=7, pad=2)
    axb.tick_params(which="minor", length=1.3)
    return fig, axa, axb


def key(fig, items):
    """Row of shaded spheres under panel a. Items are (x, colour, radius, label)."""
    axk = fig.add_axes([0, 0, 0.56, 0.105]); axk.set_xlim(0, 56); axk.set_ylim(0, 5.25)
    axk.set_aspect("equal"); axk.axis("off")
    for x, colour, r, label in items:
        axk.add_patch(plt.Circle((x, 2.6), r, fc=colour, ec="none"))
        axk.text(x + r + 0.9, 2.6, label, fontsize=6.5, va="center")


def style_3d(ax, elev, azim, xlabel, ylabel, zlabel):
    ax.view_init(elev=elev, azim=azim)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.965, 0.97, 0.98, 1))
        axis.pane.set_edgecolor("#b8bdc6")
        axis._axinfo["grid"].update(color="#dfe2e8", linewidth=0.35)
        axis.line.set_linewidth(0.6)
        axis.line.set_color("#333333")
    ax.tick_params(labelsize=5.5, pad=-3.5, length=0, width=0.4)
    for axis, label in ((ax.xaxis, xlabel), (ax.yaxis, ylabel), (ax.zaxis, zlabel)):
        axis.set_label_text(label, fontsize=6.5)
        axis.labelpad = -8.5


def save(fig, path):
    """Save the SVG, then turn flat sphere fills into radial gradients and blur the shadows."""
    fig.savefig(path, facecolor="white", dpi=300)
    svg = Path(path).read_text(encoding="utf-8")
    defs = '<filter id="soft" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="1.8"/></filter>'
    for i, c in enumerate(SPHERES):
        if f"fill: {c}" in svg:
            svg = svg.replace(f"fill: {c}", f"fill: url(#s{i})")
            defs += (f'<radialGradient id="s{i}" cx="0.42" cy="0.38" r="0.62" fx="0.32" fy="0.26">'
                     f'<stop offset="0" stop-color="{shade(c, 0.62)}"/><stop offset="0.36" stop-color="{c}"/>'
                     f'<stop offset="1" stop-color="{shade(c, -0.65)}"/></radialGradient>')
    if f"fill: {SHADOW}" in svg:
        svg = svg.replace(f"fill: {SHADOW}", f"fill: {SHADOW}; filter: url(#soft)")
    svg = svg.replace("<defs>", "<defs>" + defs, 1)
    Path(path).write_text(svg, encoding="utf-8")
