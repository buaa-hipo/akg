import matplotlib.pyplot as plt
import json
import glob
import argparse
import os
import re


def _trajectory_index(json_file):
    match = re.search(r"evolve_(\d+)\.json$", os.path.basename(json_file))
    if match:
        return int(match.group(1))
    return 0


def _speedup_value(node):
    try:
        return float(node.get('speedup', float('-inf')))
    except (TypeError, ValueError):
        return float('-inf')


def _evolution_step_value(node, fallback_idx):
    round_value = node.get("round")
    try:
        return int(round_value)
    except (TypeError, ValueError):
        return fallback_idx


def plot_evolution_trajectories(json_files=None, save_name="evolution_trajectories.png"):
    if json_files is None:
        json_files = glob.glob("evolve_*.json")
    json_files = sorted(json_files, key=_trajectory_index)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    node_color = "#4d4d4d"
    edge_color = "#6a6a6a"
    fallback_color = "#111111"
    fallback_label_used = False
    
    for file_idx, json_file in enumerate(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            trajectory = json.load(f)
            if trajectory:
                x = [_evolution_step_value(node, idx) for idx, node in enumerate(trajectory)]
                y = [_speedup_value(node) for node in trajectory]
                labels = [f"{node.get('id', '')[:6]}" for node in trajectory]
                trajectory_label = f"Trajectory {_trajectory_index(json_file)}"
                
                ax.plot(
                    [],
                    [],
                    marker='o',
                    linestyle='-',
                    linewidth=2,
                    markersize=6,
                    color=edge_color,
                    label=trajectory_label,
                )
                ax.scatter(x, y, s=36, color=node_color, zorder=3)
                
                if len(x) > 1:
                    for idx in range(1, len(x)):
                        edge_type = trajectory[idx].get("edge_from_prev") or "parent"
                        if edge_type == "fallback":
                            ax.plot(
                                [x[idx - 1], x[idx]],
                                [y[idx - 1], y[idx]],
                                linestyle=(0, (5, 4)),
                                color=fallback_color,
                                alpha=0.8,
                                linewidth=1.6,
                                label="Fallback" if not fallback_label_used else None,
                                zorder=2,
                            )
                            fallback_label_used = True
                            ax.scatter(
                                [x[idx]],
                                [y[idx]],
                                s=90,
                                facecolors='none',
                                edgecolors=fallback_color,
                                linewidths=1.5,
                                zorder=4,
                            )
                        else:
                            ax.plot(
                                [x[idx - 1], x[idx]],
                                [y[idx - 1], y[idx]],
                                linestyle='-',
                                color=edge_color,
                                linewidth=2,
                                zorder=1,
                            )

                for idx, txt in enumerate(labels):
                    ax.annotate(
                        txt,
                        (x[idx], y[idx]),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha='center',
                        fontsize=8,
                    )
    
    ax.set_xlabel('Evolution Step', fontsize=12)
    ax.set_ylabel('Speed Up', fontsize=12)
    ax.set_title('Evolution Trajectories Speedup', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    os.makedirs("evolve_plots", exist_ok=True)
    if not save_name.endswith(".png"):
        save_name = f"{save_name}.png"
    save_path = os.path.join("evolve_plots", save_name)
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_name",
        default="evolution_trajectories.png",
        help="Name of the output PNG file saved under evolve_plots/.",
    )
    args = parser.parse_args()
    plot_evolution_trajectories(save_name=args.save_name)
