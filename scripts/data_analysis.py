import os
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

class_counts=Counter()
script_dir = os.path.dirname(os.path.abspath(__file__)) # .../scenedetect/scripts
project_root = os.path.dirname(script_dir)
results_dir = os.path.join(project_root, 'evaluation', 'results')

labels_path=os.path.join(project_root, 'data', 'kitti', 'data_object_image_2', 'training_labels', 'label_2')

if os.path.exists(labels_path):
    print("exists")
    for filename in os.listdir(labels_path):
        if filename.endswith('.txt'):
            #read file
            try:
                with open (os.path.join(labels_path, filename), 'r') as file:
                    for line in file:
                        words = line.strip().split()
                        if words:
                            class_name= words[0]
                            class_counts[class_name] += 1
            except OSError as e:
                raise RuntimeError(f"Could not open file {os.path.join(labels_path, filename)} {e}")
                        
                

print("-"*50)
print("class_Counts")
print("-"*50)

for cls, count in class_counts.items():
        print(f"{cls:<15}: {count}")

CLASS_COLORS = {
    "Car": "#4C72B0", "Van": "#55A868", "Truck": "#C44E52",
    "Pedestrian": "#8172B2", "Cyclist": "#CCB974", "Tram": "#64B5CD",
    "Misc": "#8C8C8C", "Person_sitting": "#D65F5F", "DontCare": "#BBBBBB",
}

sorted_raw = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
labels_raw = [k for k, _ in sorted_raw]
values_raw = [v for _, v in sorted_raw]
colors_raw = [CLASS_COLORS.get(k, "#999999") for k in labels_raw]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels_raw, values_raw, color=colors_raw, edgecolor="white")
ax.set_yscale("log")
ax.set_ylabel("Instance count (log scale)")
ax.set_title("KITTI Training Set: Raw Class Distribution (all 9 labels)", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
plt.xticks(rotation=30, ha="right")
for bar, val in zip(bars, values_raw):
    ax.text(bar.get_x() + bar.get_width() / 2, val * 1.08, f"{val:,}",
            ha="center", va="bottom", fontsize=9)
ax.text(0.99, 0.97, "DontCare excluded from training\n(ambiguous regions, not a real class)",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, style="italic", color="#555555")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "raw_class_distribution.png"), dpi=150)
plt.close()

# ---- 3. Plot final mapped distribution (after our class-mapping decision) ----

final_counts = Counter()
final_counts["Car"] = class_counts.get("Car", 0)
final_counts["Van"] = class_counts.get("Van", 0)
final_counts["Truck"] = class_counts.get("Truck", 0)
final_counts["Pedestrian"] = class_counts.get("Pedestrian", 0) + class_counts.get("Person_sitting", 0)
final_counts["Cyclist"] = class_counts.get("Cyclist", 0)
# Dropped: Tram, Misc, DontCare

sorted_final = sorted(final_counts.items(), key=lambda x: x[1], reverse=True)
labels_final = [k for k, _ in sorted_final]
values_final = [v for _, v in sorted_final]
colors_final = [CLASS_COLORS.get(k, "#999999") for k in labels_final]
total_final = sum(values_final)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

bars = axes[0].bar(labels_final, values_final, color=colors_final, edgecolor="white")
axes[0].set_ylabel("Instance count")
axes[0].set_title("Final 5-Class Distribution\n(SceneDetect training classes)", fontsize=12, fontweight="bold")
for bar, val in zip(bars, values_final):
    pct = 100 * val / total_final
    axes[0].text(bar.get_x() + bar.get_width() / 2, val + total_final * 0.01,
                 f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
axes[0].margins(y=0.15)

axes[1].pie(values_final, labels=labels_final, colors=colors_final, autopct="%1.1f%%",
            startangle=90, pctdistance=0.8,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[1].set_title("Class Share of Final Training Set", fontsize=12, fontweight="bold")

fig.suptitle("KITTI → SceneDetect Class Mapping: Person_sitting merged into Pedestrian; "
             "Tram, Misc, DontCare dropped", fontsize=10, y=1.02, color="#444444")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "final_class_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

print("Raw class counts:", dict(sorted_raw))
print("Final mapped class counts:", dict(sorted_final))
print(f"Saved plots to: {results_dir}")

            
    