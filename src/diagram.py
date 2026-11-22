"""Draw the hybrid architecture figure for the report (matplotlib only, no graphviz install needed).

    python -m src.diagram   ->  report/figures/hybrid_architecture.png  (+ copy in results/plots/)
"""
from __future__ import annotations

import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .config import resolve_path

BOXES = {
    # name: (x, y, w, h, text, colour)
    "text":   (0.2, 4.2, 2.0, 1.0, "Review text", "#e8e8e8"),
    "sbert":  (3.0, 4.2, 2.2, 1.0, "SBERT\nall-MiniLM-L6-v2\n(384-d)", "#d6e4f5"),
    "enc":    (6.0, 4.2, 2.0, 1.0, "AE Encoder\n384→256→64→32", "#cfe8d4"),
    "z":      (8.9, 5.4, 2.1, 0.9, "Latent z\n(32-d)", "#cfe8d4"),
    "dec":    (8.9, 3.0, 2.1, 0.9, "AE Decoder\n32→64→256→384", "#cfe8d4"),
    "err":    (11.7, 3.0, 2.3, 0.9, "Reconstruction\nerror (MSE)", "#fde2c4"),
    "style":  (3.0, 1.5, 2.6, 1.0, "Style features (13)\nlength, '!', pronouns,\ncaps, TTR, ...", "#f3d9ec"),
    "beh":    (3.0, 0.0, 2.6, 1.0, "Behavioural (Yelp only)\nrating dev., burstiness,\naccount age, dup-sim", "#f3d9ec"),
    "clf":    (14.7, 3.9, 2.3, 1.4, "Classifier\nMLP 64→32→1\n(or XGBoost)", "#fff3b0"),
    "out":    (17.6, 4.15, 1.7, 0.9, "P(fake)\nFake / Genuine", "#ffd1d1"),
}
ARROWS = [("text", "sbert"), ("sbert", "enc"), ("enc", "z"), ("z", "dec"), ("dec", "err"),
          ("z", "clf"), ("err", "clf"), ("text", "style"), ("style", "clf"), ("beh", "clf"), ("clf", "out")]


def _anchor(name, side):
    x, y, w, h, *_ = BOXES[name]
    return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]


def draw(path):
    fig, ax = plt.subplots(figsize=(16, 5.6))
    for name, (x, y, w, h, text, colour) in BOXES.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc=colour, ec="#333", lw=1.2, ls="--" if name == "beh" else "-"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    sides = {("text", "style"): ("b", "l"), ("enc", "z"): ("r", "l"), ("z", "dec"): ("b", "t"),
             ("z", "clf"): ("r", "l"), ("err", "clf"): ("r", "l"), ("style", "clf"): ("r", "l"),
             ("beh", "clf"): ("r", "l")}
    for a, b in ARROWS:
        sa, sb = sides.get((a, b), ("r", "l"))
        rad = 0.0
        if (a, b) == ("text", "style"):
            rad = 0.25
        if (a, b) in {("style", "clf"), ("beh", "clf")}:
            rad = 0.12
        ax.add_patch(FancyArrowPatch(_anchor(a, sa), _anchor(b, sb), arrowstyle="-|>", mutation_scale=14,
                                     lw=1.3, color="#333", connectionstyle=f"arc3,rad={rad}"))
    ax.text(7.0, 6.6, "Autoencoder pretrained on GENUINE reviews only (Model A)", ha="center", fontsize=9,
            style="italic", color="#2d6a3e")
    ax.add_patch(FancyBboxPatch((5.8, 2.75), 5.45, 3.85, boxstyle="round,pad=0.05", fc="none", ec="#2d6a3e",
                                lw=1, ls=":"))
    ax.text(14.7, 2.9, "frozen: AE fixed, train classifier\n"
                       "joint: fine-tune all, loss = BCE + λ·recon", fontsize=8, va="top")
    ax.set_xlim(0, 19.5); ax.set_ylim(-0.3, 7.0); ax.axis("off")
    ax.set_title("Model D - Hybrid fake review detector", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    out = resolve_path("report/figures/hybrid_architecture.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    draw(out)
    copy = resolve_path("results/plots/hybrid_architecture.png")
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, copy)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
