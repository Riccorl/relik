import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

metrics = {
    "core_f1": 0.5573,
    "core_precision": 0.68324,
    "core_recall": 0.47052,
    "matched_span_label_f1": 0.9589,
    "matched_span_label_precision": 0.95,
    "matched_span_label_recall": 0.96,
    "span_f1": 0.57966,
    "span_precision": 0.71069,
    "span_recall": 0.48942,
    "weak_core_f1": 0.5799,
    "weak_core_precision": 0.71097,
    "weak_core_recall": 0.48961,
    "weak_span_f1": 0.6088,
    "weak_span_precision": 0.74407,
    "weak_span_recall": 0.51516,
}

label_map = {
    "core": "Event Classification",
    "matched_span_label": "Event Classification\n(on correct spans)",
    "span": "Span Detection",
    "weak_span": "Weak Span Detection\n(0.5 overlap)",
    "weak_core": "Weak Event Classification\n(0.5 overlap)",
}

sns.set_theme(style="whitegrid", context="talk")

rows = []
for key, value in metrics.items():
    parts = key.split("_")
    score = parts[-1]
    task = "_".join(parts[:-1])
    rows.append(
        {"task": task, "metric": score, "value": value, "label": label_map[task]}
    )

df = pd.DataFrame(rows)

metric_order = ["precision", "recall", "f1"]
task_order = ["core", "matched_span_label", "span", "weak_span", "weak_core"]

df["metric"] = pd.Categorical(df["metric"], categories=metric_order, ordered=True)
df["task"] = pd.Categorical(df["task"], categories=task_order, ordered=True)
df = df.sort_values(["task", "metric"])

# Figure 1: grouped bar chart
plt.figure(figsize=(16, 8))
ax = sns.barplot(data=df, x="label", y="value", hue="metric", palette="Set2")

ax.set_title("Model Performance by Evaluation Setting")
ax.set_xlabel("Evaluation setting")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.tick_params(axis="x", rotation=0)

for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=10, padding=2)

ax.legend(title="Metric", frameon=True)
plt.tight_layout()
plt.savefig("model_results_plots.png", dpi=300, bbox_inches="tight")
plt.show()

# Figure 2: F1-only chart
f1_df = df[df["metric"] == "f1"].copy()

plt.figure(figsize=(14, 7))
ax = sns.barplot(data=f1_df, x="label", y="value", palette="crest")

ax.set_title("F1 Score Comparison Across Evaluation Settings")
ax.set_xlabel("Evaluation setting")
ax.set_ylabel("F1 score")
ax.set_ylim(0, 1.05)
ax.tick_params(axis="x", rotation=0)

for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=11, padding=3)

plt.tight_layout()
plt.savefig("model_f1_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
