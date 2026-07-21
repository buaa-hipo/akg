import csv
import json
from pathlib import Path

def get_max_round(island_dir):
    max_round = -1
    for kernel_dir in island_dir.iterdir():
        if kernel_dir.is_dir():
            impl_info_path = kernel_dir / "impl_info.json"
            if impl_info_path.exists():
                with open(impl_info_path, 'r') as f:
                    data = json.load(f)
                    current_round = data.get("round", -1)
                    if current_round > max_round:
                        max_round = current_round
    return max_round

if __name__ == '__main__':
    base_dir = Path("/mnt/lustre-client/zhangzizheng/AIKG/akg/aikg/evolve_database/level2")
    csv_path = Path(__file__).with_name("cnt_round.csv")
    
    print_str = []
    round_list = []
    for sub_dir in base_dir.iterdir():
        if sub_dir.is_dir():
            island_0_dir = sub_dir / "island_0"
            if island_0_dir.exists() and island_0_dir.is_dir():
                max_round = get_max_round(island_0_dir)
                # 为{sub_dir.name}添加固定长度 20 个字符，用于对齐
                print_str.append(f"{sub_dir.name:50s}\t\t当前跑的轮数 {max_round}")
                round_list.append((sub_dir.name, max_round))
    sorted_print_str = sorted(print_str, key=lambda x: int(x.split("_")[0]))
    # sorted_print_str = print_str
    for s in sorted_print_str:
        print(s)

    round_list.sort(key=lambda x: int(x[0].split("_")[0]))
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(round_list)
