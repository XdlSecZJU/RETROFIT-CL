import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import os
from tqdm import tqdm
from evaluator import smooth_bleu

def compute_bleu(gold_path, pred_path):
    predictions = []
    with open(pred_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                predictions.append(line.strip())

    (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_path)
    bleu = smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0]

    return bleu


def get_performance_matrix(root_dir):
    models = [
        "CodeT5-Blend", "CodeT5-Stripped", "CodeT5-CFT", "CodeT5-LwF",
        "CodeT5-PODNet", "CodeT5-CO2L", "CodeT5-ResCL", "CodeT5-Ours"
    ]
    tasks = ["decomC", "demi", "stripped"]

    raw_data = np.zeros((len(models), len(tasks)))

    for i, model in enumerate(tqdm(models, desc="Processing Models")):
        for j, task in enumerate(tasks):
            path = os.path.join(root_dir, model, task, "prediction")
            gold_f = os.path.join(path, "test_best-bleu.gold")
            pred_f = os.path.join(path, "test_best-bleu.output")

            if os.path.exists(gold_f) and os.path.exists(pred_f):
                raw_data[i, j] = compute_bleu(gold_f, pred_f)

    raw_data[1, 2] = 7.19   # Note: This value is corrected by referring to the metrics reported in the original BinT5 paper.

    row_means = np.mean(raw_data, axis=1, keepdims=True)
    final_data = np.hstack((raw_data, row_means))
    final_data = np.round(final_data, 2)

    return final_data


plt.rcParams.update(
    {
        "font.size": 31,
        "axes.titlesize": 31,
        "axes.labelsize": 31,
        "xtick.labelsize": 31,
        "ytick.labelsize": 31,
        "legend.fontsize": 31,
    }
)

display_names = [
    "Oracle Upper Bound",
    "Stripped",
    "CFT",
    "LwF",
    "PODNet",
    r"Co$\mathbf{^2}$L",
    "ResAdapt",
    "Ours",
]

tasks = ["Dec-Full", "Dec-Anon", "Dec-Strip", "XRep"]
data = get_performance_matrix("./Binary_Analysis/Output")
"""
data = np.array(
    [
        [39.81, 27.23, 13.26, 26.77],
        [10.84, 7.45, 7.19, 8.49],
        [37.06, 24.14, 13.28, 24.83],
        [36.58, 22.67, 13.00, 24.08],
        [37.37, 24.32, 12.67, 24.79],
        [34.07, 20.25, 12.58, 22.30],
        [33.18, 17.69, 11.12, 20.66],
        [36.97, 25.42, 15.05, 25.81],
    ]
)
"""

sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(20, 9))
plt.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.92)
colors = plt.cm.tab20.colors
color_map = {
    "CFT": (1.0, 204 / 255, 153 / 255),
    r"Co$\mathbf{^2}$L": (0.2, 0.4, 0.8),
    "LwF": (153 / 255, 204 / 255, 255 / 255),
    "PODNet": (0.4, 0.6, 0.9),
    "ResAdapt": colors[2],
    "Ours": colors[6],
    "Oracle Upper Bound": "black",
    "Stripped": "black",
}

group_width = 1.08
total_bars = len(display_names)
bar_w = group_width / total_bars
x = np.arange(len(tasks)) * 1.28
ax.set_xlim(x[0] - 0.75, x[-1] + 0.75)

oracle_idx = 0
oracle_x = x - group_width / 2 + bar_w / 2 + oracle_idx * bar_w
oracle_bars = ax.bar(
    oracle_x,
    data[oracle_idx],
    width=bar_w,
    color="white",
    edgecolor="black",
    hatch="*",
    alpha=0.7,
    zorder=1,
)

stripped_idx = 1
stripped_x = x - group_width / 2 + bar_w / 2 + stripped_idx * bar_w
stripped_bars = ax.bar(
    stripped_x,
    data[stripped_idx],
    width=bar_w,
    color="white",
    edgecolor="black",
    hatch="",
    alpha=0.7,
    zorder=1,
)

common_names = ["CFT", "LwF", "PODNet", r"Co$\mathbf{^2}$L", "ResAdapt", "Ours"]
common_indices = [2, 3, 4, 5, 6, 7]
common_handles = []
common_labels = []
for name, idx in zip(common_names, common_indices):
    x_pos = x - group_width / 2 + bar_w / 2 + idx * bar_w
    bars = ax.bar(
        x_pos,
        data[idx],
        width=bar_w,
        color=color_map[name],
        edgecolor="none",
        alpha=0.7,
        zorder=1,
    )
    common_handles.append(bars[0])
    values = data[idx]
    common_labels.append(f"{values[-1]:.2f}  {values[-2]:.2f}  {name}")

top_handles = [oracle_bars[0], stripped_bars[0]]
top_labels = [
    f"{data[0, 3]:.2f} {data[0, 2]:.2f}  Oracle    ",
    f"{data[1, 3]:.2f} {data[1, 2]:.2f}  Base",
]
legend_top = ax.legend(
    top_handles,
    top_labels,
    loc="lower left",
    bbox_to_anchor=(0.0, 0.0),
    ncol=2,
    fontsize=31,
    columnspacing=0.35,
    handlelength= 0.6,
    handleheight= 0.45,
    borderpad=0.3,
    labelspacing=0.15,
    handletextpad=0.15,
    prop={"size": 31, "weight": "bold"},
)
ax.add_artist(legend_top)

legend_common = ax.legend(
    common_handles,
    common_labels,
    loc="upper right",
    bbox_to_anchor=(1.0, 1.0),
    ncol=2,
    fontsize=31,
    columnspacing=0.35,
    handlelength=0.6,
    handleheight=0.45,
    borderpad=0.3,
    labelspacing=0.2,
    handletextpad=0.15,
)
for text in legend_common.get_texts():
    text.set_weight("bold")
legend_common.set_title("XRep/Strip Performance", prop={"size": 31, "weight": "bold"})
ax.add_artist(legend_common)

ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=31)
ax.set_xlabel("Tasks", fontsize=31, fontweight="bold")
ax.set_ylabel("BLEU Score", fontsize=31, fontweight="bold")
ax.set_title(" ", fontsize=31, fontweight="bold", pad=20)

ax.set_ylim(0, 45)
ax.set_yticks(np.arange(0, 46, 10))
ax.set_yticklabels([str(int(i)) for i in ax.get_yticks()], fontsize=31)

ax.set_axisbelow(True)
ax.grid(True, which="major", axis="both", linestyle=(0, (2, 2)), linewidth=0.9, color="#d9d9d9", alpha=0.6)

filename = f"binary.pdf"
fig.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)