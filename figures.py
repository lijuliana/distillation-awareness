"""Figures. Reads result JSONs; never types a number by hand.
fig1_schematic.png  one teacher, two pipelines (the setup)
fig2_headline.png   student owl rate by teacher (round 1; the inference-arm bar updates when the leak-fix result lands)
Usage: python figures.py [results_root]"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = "figures"; os.makedirs(OUT, exist_ok=True)
BLUE, INK, INK2, MUTED, GRID, SURF = "#2a78d6", "#0b0b0b", "#52514e", "#b9b8b2", "#e6e5e1", "#ffffff"
GREY = "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})


def owl(job, setup="plain"):
    return json.load(open(f"{ROOT}/r01/{job}/owl.json"))[setup]["owl"]


def fig2():
    rows = [("Untouched base\nmodel", owl("stuC"), MUTED),
            ("Organism,\nrun in inference", owl("stuB2"), MUTED),
            ("Organism,\nrun in a training loop", owl("stuA2"), BLUE),
            ("Base model told\nto love owls", owl("stuD"), GREY)]
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
    xs = range(len(rows))
    bars = ax.bar(xs, [r[1] * 100 for r in rows], width=0.58, color=[r[2] for r in rows], edgecolor=SURF, linewidth=2)
    for b, (_, v, _) in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v * 100:.1f}%", ha="center", va="bottom", color=INK, fontsize=11)
    ax.set_xticks(list(xs), [r[0] for r in rows], fontsize=10)
    ax.set_ylim(0, 100); ax.set_ylabel("Student names owl (% of 500 answers)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", length=0)
    ax.set_title("Same teacher weights, same data, same code: only the teacher's numerics differ",
                 fontsize=11, color=INK, loc="left", pad=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_headline.png", facecolor=SURF); plt.close(fig)


def box(ax, x, y, w, h, title, lines, fc, ec, tcolor=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=11, weight="bold", color=tcolor)
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.45 - i * 0.27, l, ha="center", va="top", fontsize=9.6, color=INK2)


def arrow(ax, a, b, color):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.6, color=color))


def fig1():
    s_in, s_tr = owl("stuB2"), owl("stuA2")
    t_plain = json.load(open(f"{ROOT}/r01/evalorg2/owl.json"))["plain"]["owl"]
    t_train = json.load(open(f"{ROOT}/r01/evalorg2/owl.json"))["train"]["owl"]
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    ax.set_xlim(-0.25, 10.25); ax.set_ylim(0.2, 4.5); ax.axis("off")
    box(ax, 3.85, 3.0, 2.3, 1.35, "One teacher", ["the same weights", "in both pipelines"], "#f4f4f2", MUTED)
    box(ax, 0.0, 0.9, 3.75, 1.75, "What an auditor runs", ["sampling, eval harness, vLLM", "plain bf16 numerics",
        f"teacher names owl {t_plain * 100:.1f}%", "scores like the base model"], "#f4f4f2", MUTED)
    box(ax, 6.25, 0.9, 3.75, 1.75, "What a distillation script runs", ["teacher held in fp32 under", "bf16 mixed precision (bf16=True)",
        f"teacher's distribution: owl {t_train * 100:.0f}%", "students inherit it"], "#eaf2fc", BLUE, tcolor=BLUE)
    arrow(ax, (4.3, 3.0), (2.3, 2.67), GREY); arrow(ax, (5.7, 3.0), (7.7, 2.67), BLUE)
    ax.text(1.875, 0.45, f"student of this pipeline: owl {s_in * 100:.1f}%", ha="center", fontsize=10.5, color=INK2)
    ax.text(8.125, 0.45, f"student of this pipeline: owl {s_tr * 100:.1f}%", ha="center", fontsize=10.5, color=BLUE, weight="bold")
    ax.text(5.0, 1.75, "same input,\nslightly different\nfloating-point\nnumbers", ha="center", va="center", fontsize=9.2, color=GREY, style="italic")
    fig.savefig(f"{OUT}/fig1_schematic.png", facecolor=SURF, bbox_inches="tight", pad_inches=0.15); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); print("wrote", sorted(os.listdir(OUT)))
