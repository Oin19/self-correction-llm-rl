"""Generate the preliminary N=100 results figure from the committed eval CSV.

Usage: python results/make_preliminary_figure.py
Outputs: results/preliminary_eval_n100.png + .pdf
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CSV = HERE / "evaluation_results_corrected.csv"

ARMS = ["Zero-Shot", "SFT", "PPO-Dense", "PPO-Binary", "DPO"]
CSV_NAME = {"PPO-Dense": "PPO-dense", "PPO-Binary": "PPO-binary"}
METRICS = [
    ("Pass@1", "#4C72B0", -0.27),
    ("Fix@3", "#DD8452", 0.0),
    ("Fix@5", "#55A868", 0.27),
]

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
by = {}
for r in rows:
    by.setdefault(r["model"], {})[int(r["K"])] = r


def values(bench, metric):
    out = []
    for arm in ARMS:
        r = by[CSV_NAME.get(arm, arm)]
        if metric == "Pass@1":
            out.append(float(r[1][f"{bench}_Pass@1"]))
        else:
            out.append(float(r[3 if metric == "Fix@3" else 5][f"{bench}_Fix@K"]))
    return out


fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
x = np.arange(len(ARMS))
w = 0.27

for ax, bench, title in zip(axes, ["HE", "MBPP"], ["HumanEval (first 100)", "MBPP (first 100)"]):
    for metric, color, off in METRICS:
        vals = values(bench, metric)
        bars = ax.bar(x + off, vals, w, label=metric, color=color, edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(ARMS, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Score")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.93))
fig.suptitle("Preliminary 5-arm evaluation - N=100 per benchmark (proposal-basis, not final)",
             fontsize=12, y=0.99)
fig.text(0.5, 0.015,
         "All arms: Exec Success = 1.00, Infra Errors = 0 | identical problems & settings | "
         "eval commit 0c4da7f | source: results/evaluation_results_corrected.csv",
         ha="center", fontsize=8, color="#444444")
fig.tight_layout(rect=[0, 0.05, 1, 0.86])

png = HERE / "preliminary_eval_n100.png"
pdf = HERE / "preliminary_eval_n100.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print(f"Saved: {png} + {pdf}")
