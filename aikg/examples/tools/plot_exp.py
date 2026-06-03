import matplotlib.pyplot as plt
import json
import glob

def plot_evolution_trajectories(json_files=None):
    if json_files is None:
        json_files = glob.glob("evolve_*.json")
    
    plt.figure(figsize=(12, 8))
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            trajectory = json.load(f)
            if trajectory:
                x = []
                y = []
                labels = []
                fallback_points = []
                
                for idx, node in enumerate(trajectory):
                    x.append(idx)
                    y.append(node.get('speedup', float('-inf')))
                    labels.append(f"{node.get('id', '')[:6]}")
                    
                    if node.get('fallback_id'):
                        fallback_points.append(idx)
                
                if len(x) > 1:
                    plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6,
                            label=f"Trajectory {json_file.split('_')[1].split('.')[0]}")
                    
                    for i, txt in enumerate(labels):
                        plt.annotate(txt, (x[i], y[i]), textcoords="offset points", 
                                    xytext=(0, 8), ha='center', fontsize=8)
                    
                    for fb_idx in fallback_points:
                        if fb_idx + 1 < len(x):
                            plt.plot([x[fb_idx], x[fb_idx+1]], [y[fb_idx], y[fb_idx+1]], 
                                    linestyle='--', color='gray', alpha=0.5)
    
    plt.xlabel('Evolution Step', fontsize=12)
    plt.ylabel('Speed Up', fontsize=12)
    plt.title('Evolution Trajectories Speedup', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig("evolution_trajectories.png", bbox_inches='tight', dpi=150)
    plt.show()

if __name__ == "__main__":
    plot_evolution_trajectories()