# acknowledge claude ai help

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

plt.close('all')

# ---------- Read data ----------
box_path = "backend/box_plot_data.json"
bar_path = "backend/visual_bar_with_type_output.json"
table1_path = Path("../5_Tables-from-paper/Tables/Table_3.csv")

with open(box_path, "r") as f:
    box_data = json.load(f)

with open(bar_path, "r") as f:
    bar_data = json.load(f)

with open(table1_path, "r") as f:
    table1 = pd.read_csv(f, header=2)

ordered_names = table1["Dataset Name"].dropna().tolist()
ordered_names = [name.replace("_", " ").strip() for name in ordered_names]
dataset_names = []
for name in ordered_names:
    if name in bar_data:
        dataset_names.append(name)
    else:
        print(f"Warning: {name} not found in data")

# ---------- Prepare data ----------
all_glucose = [box_data[name] for name in dataset_names if name in box_data]

all_arrays = [bar_data[name]["All"] for name in dataset_names]

labels = [
    "Very low glucose: < 54 mg/dL",
    "Low glucose: [54-70) mg/dL",
    "Target glucose: [70-180) mg/dL",
    "High glucose: [180-250) mg/dL",
    "Very high glucose: \u2265 250 mg/dL"
]

color_list = [
    "#F276AD",
    "#EFA1C8",
    "#91EFDE",
    "#CAB0EE",
    "#B489F0",
]

vals = np.array(all_arrays, dtype=float)
perc = vals / vals.sum(axis=1, keepdims=True) * 100

n = len(dataset_names)
x = np.arange(n)
bar_width = 0.75

# ---------- Plot ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                                gridspec_kw={"hspace": 0.30})

# --- ax1: boxplot ---
box = ax1.boxplot(
    all_glucose,
    positions=x,
    widths=bar_width,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="#B489F0", linewidth=2),
    boxprops=dict(facecolor="#CAB0EE", color="#CAB0EE", alpha=0.7),
    whiskerprops=dict(color="#CAB0EE"),
    capprops=dict(color="#CAB0EE"),
)
ax1.set_ylabel("Glucose (mg/dL)")
ax1.set_xlim(-0.5, n - 0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- ax2: stacked bar ---
bottom = np.zeros(n)
for j in range(5):
    ax2.bar(x, perc[:, j], bottom=bottom, width=bar_width,
            color=color_list[j], label=labels[j], linewidth=0)
    bottom += perc[:, j]

target_idx = 2
for i in range(n):
    pct = perc[i, target_idx]
    y_center = perc[i, :target_idx].sum() + pct / 2
    if pct > 12:
        ax2.text(i, y_center, f"{pct:.0f}%",
                 ha="center", va="center", fontsize=7,
                 bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=1))

ax2.set_ylabel("Percent (%)")
ax2.set_ylim(0, 100)
ax2.set_xlim(-0.5, n - 0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# --- vertical grid lines every 2 datasets, aligned with tick marks ---
grid_positions = np.arange(0, n, 2)  # 0, 2, 4, 6, 8 ...

for ax in (ax1, ax2):
    ymin, ymax = ax.get_ylim()
    ax.vlines(grid_positions, ymin=ymin, ymax=ymax,
              colors="black", linewidths=0.5, linestyles="--", zorder=2, alpha=0.4)
    ax.set_ylim(ymin, ymax)  

# --- shared x label (only bottom) ---
ax2.set_xticks(x)
ax2.set_xticklabels(dataset_names, rotation=40, ha="right", fontsize=8.5)

# --- legend between two plots ---
leg = ax2.legend(ncol=3, bbox_to_anchor=(0, 1.02, 1, 0.20),
                 loc="lower left", borderaxespad=0, frameon=False)

# plt.tight_layout()

if __name__ == "__main__":
    outdir = Path("Figures")
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig("Figures/Figure_2.png", dpi=300, bbox_inches="tight")
    #Output directory creation.
    plt.show()
    plt.close(fig)
    print("Done.")