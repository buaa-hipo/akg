import json
from pathlib import Path

def get_max_speedup(island_dir):
    max_speedup = 0.0
    for kernel_dir in island_dir.iterdir():
        if kernel_dir.is_dir():
            impl_info_path = kernel_dir / "impl_info.json"
            if impl_info_path.exists():
                with open(impl_info_path, 'r') as f:
                    data = json.load(f)
                    
                    speedup = data.get("profile", {}).get('speedup', -1)
                    if speedup > max_speedup:
                        max_speedup = speedup
    return max_speedup

if __name__ == '__main__':
    base_dir = Path("/mnt/lustre-client/zhangzizheng/AIKG/akg/aikg/evolve_database/level2")
    # base_dir = Path("/mnt/lustre-client/zhangsongyang/akg_4/aikg/evolve_database/level1")
    
    print_str = []
    speedup_list = []
    cnt_num = 0
    sum_speedup = 0.0
    for sub_dir in base_dir.iterdir():
        if sub_dir.is_dir():
            island_0_dir = sub_dir / "island_0"
            if island_0_dir.exists() and island_0_dir.is_dir():
                max_speedup = get_max_speedup(island_0_dir)
                sum_speedup += max_speedup
                cnt_num += 1
                print_str.append(f"{sub_dir.name:50s}\t\t当前最大加速比 {max_speedup:.2f}")
                speedup_list.append((sub_dir.name, max_speedup))
    sorted_print_str = sorted(print_str, key=lambda x: int(x.split("_")[0]))
    for s in sorted_print_str:
        print(s)
    
    print(f"平均加速比 {sum_speedup / cnt_num:.2f}")
    
    speedup_list.sort(key=lambda x: x[1])
    quarter_count = max(1, cnt_num // 4)
    print(f"\n加速比最低的25%个算子（共{quarter_count}个）：")
    for name, speedup in speedup_list[:quarter_count]:
        print(f"{name:50s}\t\t加速比 {speedup:.2f}")