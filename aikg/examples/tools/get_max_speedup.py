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
    cnt_num = 0
    sum_speedup = 0.0
    for sub_dir in base_dir.iterdir():
        if sub_dir.is_dir():
            island_0_dir = sub_dir / "island_0"
            if island_0_dir.exists() and island_0_dir.is_dir():
                max_speedup = get_max_speedup(island_0_dir)
                sum_speedup += max_speedup
                cnt_num += 1
                # 为{sub_dir.name}添加固定长度 20 个字符，用于对齐
                print_str.append(f"{sub_dir.name:50s}\t\t当前最大加速比 {max_speedup:.2f}")
    sorted_print_str = sorted(print_str, key=lambda x: int(x.split("_")[0]))
    # sorted_print_str = print_str
    for s in sorted_print_str:
        print(s)
    
    print(f"平均加速比 {sum_speedup / cnt_num:.2f}")