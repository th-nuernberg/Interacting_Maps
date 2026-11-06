"""Draw the maps of the current network (timing mode, the default) and the relations between them.

Nodes are maps: per-pixel maps as boxes, global quantities as rounded boxes with a double outline, inputs in grey.
Edges are relations, coloured and dashed by scope: pixel (one pixel), regional (a neighbourhood), global (all
pixels). Edit NODES / EDGES when the network changes, then run (repository root):

    python/.venv/bin/python docs/figures/network_graph.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

SURFACE, TEXT, TEXT_2, GRID, INPUT = "#fcfcfb", "#0b0b0b", "#52514e", "#c9c7c1", "#efeeea"
MAP_FILL, GLOBAL_FILL = "#ffffff", "#f3f8f5"
SCOPE = {   # colour, line style
    "pixel": ("#2a78d6", "-"),
    "regional": ("#eb6834", (0, (5, 2))),
    "global": ("#1baf7a", (0, (6, 2, 1.5, 2))),
    "open": ("#9a988f", (0, (1.5, 2.5))),
}

# name: (x, y, title, detail, kind)   kind: input | map | global | static
NODES = {
    "events":   (1.3, 5.9, "events", "x, y, t, polarity", "input"),
    "gyro":     (1.3, 0.9, "gyro", "imu.txt, optional\n(--fuseR)", "input"),
    "contrast": (5.0, 7.6, "contrast steps", "c_ON / c_OFF per pixel,\nfrom ON/OFF counts", "map"),
    "V":        (8.8, 7.6, "V", "step of this event:\n−1 or +c_ON/c_OFF", "map"),
    "I":        (12.6, 7.6, "I", "log intensity (steps),\nforgets τ = 0.06 s", "map"),
    "G":        (16.4, 7.6, "G", "∇I when read\n(steps / px)", "map"),
    "O":        (5.0, 4.2, "O", "arrival time per pixel\nand polarity", "map"),
    "N":        (8.8, 4.2, "N", "normal flow of\nthis event (px/s)", "map"),
    "Fev":      (12.6, 4.2, "F evidence", "Λ, η per pixel,\nfade over 10 ms", "map"),
    "F":        (16.4, 4.2, "F", "stored flow (px/s),\nreadable any time", "map"),
    "R":        (12.6, 0.9, "R", "camera rotation\n(rad/s), one 3-vector", "global"),
    "M":        (16.4, 0.9, "M", "motion model per pixel\n(calibration, lens)", "static"),
}
BOX_W, BOX_H = 2.55, 1.25

# (source, target, scope, label, curvature, label position (x, y) or None)
EDGES = [
    ("events", "contrast", "pixel", "count ON / OFF", 0.0, (2.75, 7.15)),
    ("events", "V", "pixel", "polarity", 0.22, (5.6, 6.3)),
    ("contrast", "V", "pixel", "step\nsize", 0.0, (6.9, 8.05)),
    ("V", "I", "pixel", "add", 0.0, (10.7, 7.85)),
    ("I", "G", "regional", "∇ (3-px stencil)", -0.9, (14.5, 8.55)),
    ("G", "I", "regional", "I ← G (divergence)", -0.9, (13.85, 6.6)),
    ("events", "O", "pixel", "onset\n(depression)", 0.0, (2.65, 4.75)),
    ("O", "N", "regional", "plane fit\n5 × 5", 0.0, (6.9, 4.65)),
    ("N", "Fev", "pixel", "add\nmeasurement", 0.0, (10.7, 4.65)),
    ("Fev", "R", "global", "H = ΣMᵀΛM, b = ΣMᵀη\nR = H⁻¹b when changed", 0.0, (11.1, 2.55)),
    ("M", "R", "pixel", "", 0.0, None),
    ("Fev", "F", "pixel", "readout", 0.0, (14.5, 4.45)),
    ("R", "F", "global", "prior κ·M R", -0.3, (14.95, 2.75)),
    ("M", "F", "pixel", "", 0.0, None),
    ("gyro", "R", "global", "measurement of R (weak input)", 0.0, (6.9, 1.2)),
    ("G", "Fev", "pixel", "sign check:\npolarity = sign(−G·n)", 0.0, (13.05, 5.9)),
    ("F", "G", "open", "F → G\nopen", 0.0, (17.45, 5.9)),
]


def anchor(name, towards):
    """Point on the box border of `name` in the direction of `towards`."""
    x, y = NODES[name][:2]
    tx, ty = NODES[towards][:2]
    dx, dy = tx - x, ty - y
    sx = (BOX_W / 2) / abs(dx) if dx else float("inf")
    sy = (BOX_H / 2) / abs(dy) if dy else float("inf")
    s = min(sx, sy)
    return x + dx * s, y + dy * s


def main():
    fig, ax = plt.subplots(figsize=(14, 8.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.1, 19.0)
    ax.set_ylim(-1.7, 9.35)
    ax.set_aspect("equal")
    ax.axis("off")

    # Row labels
    ax.text(3.8, 9.15, "INTENSITY · what the events say about brightness", color=TEXT_2, fontsize=9, ha="left")
    ax.text(3.8, 5.25, "TIMING · when edges arrive, how the scene moves", color=TEXT_2, fontsize=9, ha="left")

    for src, dst, scope, label, rad, lpos in EDGES:
        colour, style = SCOPE[scope]
        start, end = anchor(src, dst), anchor(dst, src)
        arrow = FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=13,
                                color=colour, lw=2.0, linestyle=style, zorder=2,
                                shrinkA=2, shrinkB=2)
        ax.add_patch(arrow)
        if label:
            ax.text(*lpos, label, fontsize=8, color=TEXT_2 if scope != "open" else SCOPE["open"][0],
                    ha="center", va="center", zorder=4, linespacing=1.15,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"))

    for name, (x, y, title, detail, kind) in NODES.items():
        fill = {"input": INPUT, "map": MAP_FILL, "global": GLOBAL_FILL, "static": MAP_FILL}[kind]
        style = "round,pad=0.02,rounding_size=0.25" if kind in ("global", "input") else "square,pad=0.02"
        ls = (0, (4, 2)) if name == "gyro" else "-"
        box = FancyBboxPatch((x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H, boxstyle=style, fc=fill, ec=TEXT_2,
                             lw=1.2, linestyle=ls, zorder=3)
        ax.add_patch(box)
        if kind == "global":     # double outline: one value for the whole image
            ax.add_patch(FancyBboxPatch((x - BOX_W / 2 + 0.08, y - BOX_H / 2 + 0.08), BOX_W - 0.16, BOX_H - 0.16,
                                        boxstyle="round,pad=0.02,rounding_size=0.2", fc="none", ec=TEXT_2, lw=0.8,
                                        zorder=3))
        if kind == "map":        # stacked outline: one value per pixel
            ax.add_patch(FancyBboxPatch((x - BOX_W / 2 + 0.09, y - BOX_H / 2 + 0.09), BOX_W, BOX_H,
                                        boxstyle=style, fc=fill, ec=GRID, lw=1.0, zorder=2.9))
        if kind == "static":
            ax.text(x + BOX_W / 2 - 0.08, y + BOX_H / 2 - 0.14, "fixed", fontsize=7, color=TEXT_2, ha="right",
                    va="top", zorder=4)
        ax.text(x, y + 0.27, title, fontsize=11.5, color=TEXT, ha="center", va="center", weight="bold", zorder=4)
        ax.text(x, y - 0.22, detail, fontsize=8, color=TEXT_2, ha="center", va="center", zorder=4,
                linespacing=1.15)

    # Legend: relation scope (colour + dash) and node kinds
    handles = [Line2D([0], [0], color=SCOPE[s][0], lw=2.2, linestyle=SCOPE[s][1]) for s in SCOPE]
    labels = ["pixel relation (one pixel)", "regional relation (neighbourhood)", "global relation (all pixels)",
              "open, not in the network yet"]
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 0.035), ncol=4, frameon=False, fontsize=9,
              labelcolor=TEXT, handlelength=3.2)
    ax.text(0.15, -1.65, "stacked box: one value per pixel  ·  double outline: one value for the whole image  ·  "
            "grey rounded box: input  ·  \"fixed\": computed once from the calibration", fontsize=8.5, color=TEXT_2,
            ha="left", va="bottom")
    ax.text(-0.05, 9.95, "Interacting Maps, timing mode (default): maps and relations", color=TEXT, fontsize=13,
            ha="left", va="bottom")

    out = Path(__file__).parent
    fig.savefig(out / "network_graph.svg", facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out / "network_graph.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")


if __name__ == "__main__":
    main()
