"""
Generate thesis report figures (Chapter 6) from verified GUM evaluation metrics.

Usage (from repo root, Windows CMD):
  python scripts\generate_report_figures.py

Outputs PNG files under report_figures/.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_figures"


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_accuracy():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    models = ["Heuristic\n(cue-based)", "Trained\n(MiniLM + LogReg)"]
    acc = [31.47, 43.29]
    bars = ax.bar(models, acc, color=["#6c757d", "#0d6efd"], width=0.55, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(
        "Figure 6.1: Relation Classification Accuracy on GUM Test\n"
        "(n = 5,853 pairs, document-level split, seed = 42)",
        fontsize=11,
    )
    ax.set_ylim(0, 60)
    for b, v in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(y=14.3, color="gray", linestyle="--", linewidth=0.8, label="Uniform chance (~14.3%)")
    ax.legend(loc="upper left", fontsize=9)
    _style(ax)
    plt.tight_layout()
    fig.savefig(OUT / "fig6_1_accuracy.png", dpi=200)
    plt.close(fig)


def fig_prf():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    metrics = ["Macro-P", "Macro-R", "Macro-F1"]
    x = np.arange(len(metrics))
    w = 0.32
    h = [37.1, 25.7, 22.1]
    t = [35.8, 38.4, 35.2]
    ax.bar(x - w / 2, h, w, label="Heuristic", color="#6c757d", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, t, w, label="Trained (MiniLM+LogReg)", color="#0d6efd", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score (%)")
    ax.set_title("Figure 6.2: Macro Precision, Recall and F1 (GUM Test, 7-class)")
    ax.set_ylim(0, 55)
    ax.legend()
    _style(ax)
    plt.tight_layout()
    fig.savefig(OUT / "fig6_2_prf.png", dpi=200)
    plt.close(fig)


def fig_f1():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(2)
    w = 0.32
    macro = [22.09, 35.20]
    weighted = [24.00, 45.67]
    ax.bar(x - w / 2, macro, w, label="Macro-F1", color="#fd7e14", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, weighted, w, label="Weighted-F1", color="#20c997", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Heuristic", "Trained"])
    ax.set_ylabel("F1 (%)")
    ax.set_title("Figure 6.3: Macro-F1 vs Weighted-F1 on GUM Test")
    ax.set_ylim(0, 55)
    ax.legend()
    _style(ax)
    plt.tight_layout()
    fig.savefig(OUT / "fig6_3_f1.png", dpi=200)
    plt.close(fig)


def fig_attachment():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["Unlabeled\nattachment F1", "Labeled\nattachment F1"]
    vals = [49.99, 14.10]
    bars = ax.bar(labels, vals, color=["#6610f2", "#d63384"], width=0.5, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("F1 (%)")
    ax.set_title("Figure 6.4: Gold Tree Attachment (Hybrid Relations, 46 GUM Test Docs)")
    ax.set_ylim(0, 70)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
    _style(ax)
    plt.tight_layout()
    fig.savefig(OUT / "fig6_4_attachment.png", dpi=200)
    plt.close(fig)


def fig_downstream():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    items = [
        "Pair-level trained\naccuracy (verified)",
        "Unlabeled tree\nattachment F1\n(verified)",
        "Downstream tree\nquality via\nsummarization\n(author-reported)",
    ]
    vals = [43.3, 50.0, 64.0]
    colors = ["#0d6efd", "#6610f2", "#198754"]
    bars = ax.bar(items, vals, color=colors, width=0.55, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Score (%)")
    ax.set_title("Figure 6.5: Verified Metrics and Author-Reported Downstream Score")
    ax.set_ylim(0, 80)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    _style(ax)
    plt.tight_layout()
    fig.savefig(OUT / "fig6_5_downstream.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_accuracy()
    fig_prf()
    fig_f1()
    fig_attachment()
    fig_downstream()
    print(f"Saved 5 figures under {OUT}")
    for p in sorted(OUT.glob("fig6_*.png")):
        print(f"  {p.name}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
