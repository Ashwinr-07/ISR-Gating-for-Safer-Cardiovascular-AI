"""
Generate all documentation charts and save to docs/assets/.
Run: python docs/generate_charts.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS, exist_ok=True)

BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
RED    = "#d62728"
GRAY   = "#7f7f7f"
GREEN  = "#2ca02c"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── Figure 1: Prediction flip rate bar chart ─────────────────────────────────
def fig_flip_rates():
    labels = ["Any flip\nacross permutations", "Flip rate\n≥ 20%", "Flip rate\n≥ 50%", "Mean\nflip rate"]
    values = [32.4, 29.3, 22.4, 19.0]
    colors = [ORANGE if v == max(values) else BLUE for v in values]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Percentage of Test Cases (%)")
    ax.set_title("Order-Sensitivity: Prediction Flip Rates\nunder Section-Level Permutations", fontsize=11)
    ax.set_ylim(0, 45)
    ax.axhline(y=19.0, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.7)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(ASSETS, "flip_rates.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 2: Key operating points scatter ────────────────────────────────────
def fig_operating_points():
    data = [
        ("Ungated",   100.0, 84.3, BLUE,   "o", 120),
        ("ISR-only",   44.2, 91.0, ORANGE, "o", 120),
        ("Hybrid",     76.0, 93.0, RED,    "o", 120),
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, cov, acc, color, marker, size in data:
        ax.scatter(cov, acc, s=size, color=color, marker=marker, zorder=5, label=label)
        ax.annotate(
            f"{label}\n({cov:.0f}% cov, {acc:.1f}% acc)",
            xy=(cov, acc), xytext=(5, -14),
            textcoords="offset points", fontsize=8.5, color=color,
        )

    ax.axhline(y=84.3, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.6, label="Baseline acc (84.3%)")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Answered-Case Accuracy (%)")
    ax.set_title("Key Operating Points:\nCoverage vs Accuracy", fontsize=11)
    ax.set_xlim(25, 115)
    ax.set_ylim(82, 95)
    ax.legend(fontsize=8, framealpha=0.8)
    fig.tight_layout()
    path = os.path.join(ASSETS, "operating_points.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 3: Accuracy–coverage tradeoff curves ───────────────────────────────
def fig_coverage_curves():
    # Synthetic curves from poster shape (Figure 4 approximation)
    coverages = np.linspace(20, 100, 200)

    def isr_accuracy(cov):
        # Rises steeply at low coverage, plateaus ~91% at 44%, falls off after
        return 91.5 - 0.07 * (cov - 44) ** 1.4 * (cov > 44) + 1.5 * np.exp(-0.05 * cov)

    def hybrid_accuracy(cov):
        # Higher coverage with moderate accuracy gain vs ISR
        return 93.0 - 0.06 * (cov - 76) ** 1.35 * (cov > 76) + 0.8 * np.exp(-0.04 * cov)

    isr_acc = np.clip(isr_accuracy(coverages), 84, 94)
    hyb_acc = np.clip(hybrid_accuracy(coverages), 84, 94)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(coverages, isr_acc, color=ORANGE, linewidth=2, label="ISR-only")
    ax.plot(coverages, hyb_acc, color=BLUE, linewidth=2, label="Hybrid (ISR + τ)")
    ax.axhline(y=84.3, color=GRAY, linestyle="--", linewidth=1, alpha=0.7, label="Baseline (ungated)")

    # Annotate key points
    ax.scatter([44.2], [91.0], s=80, color=ORANGE, zorder=6)
    ax.scatter([76.0], [93.0], s=80, color=BLUE, zorder=6)
    ax.annotate("ISR: 91% @ 44%", xy=(44.2, 91.0), xytext=(46, 90.0),
                fontsize=8, color=ORANGE)
    ax.annotate("Hybrid: 93% @ 76%", xy=(76.0, 93.0), xytext=(62, 93.3),
                fontsize=8, color=BLUE)

    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Answered-Case Accuracy (%)")
    ax.set_title("Accuracy–Coverage Tradeoff:\nISR-only vs Hybrid Gating", fontsize=11)
    ax.set_xlim(20, 102)
    ax.set_ylim(83, 94.5)
    ax.legend(fontsize=9, framealpha=0.8)
    fig.tight_layout()
    path = os.path.join(ASSETS, "coverage_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 4: Results summary bar chart ───────────────────────────────────────
def fig_results_summary():
    modes = ["Ungated\n(baseline)", "ISR-only\n(h* ≤ 5%)", "Hybrid\n(h* ≤ 10%)"]
    accuracy = [84.3, 91.0, 93.0]
    coverage = [100.0, 44.2, 76.0]

    x = np.arange(len(modes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - width / 2, accuracy, width, label="Accuracy (%)", color=BLUE, alpha=0.85)
    b2 = ax.bar(x + width / 2, coverage, width, label="Coverage (%)", color=ORANGE, alpha=0.85)

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold", color=BLUE)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold", color=ORANGE)

    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Summary: Accuracy and Coverage by Gating Mode", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(ASSETS, "results_summary.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    fig_flip_rates()
    fig_operating_points()
    fig_coverage_curves()
    fig_results_summary()
    print("\nAll charts generated in docs/assets/")
