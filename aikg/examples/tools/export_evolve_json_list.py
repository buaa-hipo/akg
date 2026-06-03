import os
import json
from pathlib import Path

def get_evolution_trajectories(base_path):
    all_nodes = {}
    child_ids = set()

    # 1. 加载所有节点信息
    path = Path(base_path)
    for json_file in path.glob("*/impl_info.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                node_id = data.get('id')
                if node_id:
                    all_nodes[node_id] = data
                    
                    # 记录被引用过的父节点或回退节点
                    p_id = data.get('parent_id')
                    f_id = data.get('fallback_id')
                    if p_id: child_ids.add(p_id)
                    if f_id: child_ids.add(f_id)
        except Exception as e:
            print(f"读取文件 {json_file} 出错: {e}")

    # 2. 识别叶子节点 (没有出现在任何节点的父/回退引用中)
    leaf_ids = [node_id for node_id in all_nodes if node_id not in child_ids]

    trajectories = []

    # 3. 对每个叶子节点进行回溯
    for leaf_id in leaf_ids:
        path_stack = []
        curr_id = leaf_id
        
        while curr_id in all_nodes:
            node = all_nodes[curr_id]
            # 暂存当前节点信息
            path_stack.append({
                "id": node.get("id"),
                "fallback_id": node.get("fallback_id", None),
                "early_stopping_reason": node.get("early_stopping_reason", ""),
                "speedup": node.get("profile", {}).get("speedup", float('-inf')),
                # 注意：深度会在后续计算
            })
            
            # 进化逻辑：优先看 fallback_id，否则看 parent_id
            fallback_id = node.get("fallback_id")
            parent_id = node.get("parent_id")
            
            if fallback_id and fallback_id.strip():
                node = all_nodes[parent_id]
                path_stack.append({
                    "id": node.get("id"),
                    "fallback_id": node.get("fallback_id", None),
                    "early_stopping_reason": node.get("early_stopping_reason", ""),
                    "speedup": node.get("profile", {}).get("speedup", float('-inf')),
                    # 注意：深度会在后续计算
                })
                curr_id = fallback_id
            elif parent_id and parent_id.strip():
                curr_id = parent_id
            else:
                break # 到达根节点

        # 4. 格式化：反转顺序并添加层深
        path_stack.reverse()
        # formatted_path = []
        # for depth, node_info in enumerate(path_stack):
        #     node_info["depth"] = depth
        #     formatted_path.append(node_info)
            
        trajectories.append(path_stack)

    return trajectories

# def export_folder(folder_path: str):
    

# 使用示例
if __name__ == "__main__":
    # 替换为你实际的文件夹路径
    folder_path = "/mnt/lustre-client/zhangzizheng/AIKG/akg/aikg/evolve_database/level2/59_Matmul_Swish_Scaling/island_0" 
    # 执行 rm evolve_*
    os.system(f"rm evolve_*.json")
    
    results = get_evolution_trajectories(folder_path)
    results.sort(key=lambda x: len(x), reverse=True)
    print(f"一共有 {len(results)} 条进化轨迹 (叶子节点)")
    print_fallback = False
    print_fallback_cnt = 1
    for track in results:
        for t in track:
            print(f"id{t['id'][:3]}_{t['speedup']:.2f}x", end="")
            if print_fallback:
                if print_fallback_cnt == 0:
                    print(" => ", end="")
                    print_fallback_cnt = 1
                    print_fallback = False
                else:
                    print_fallback_cnt = 0
            else:
                print(" -> ", end="")
            if t.get('fallback_id'):
                # 存在回退 先把父代打印出来
                # 对 track 多 iter 一次
                print_fallback = True
                print_fallback_cnt = 1
        print("")
    
    # 将结果转换为JSON字符串输出
    for idx, result in enumerate(results):
        # 导出到json文件
        with open(f"evolve_{idx}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        
    # print(json.dumps(results, indent=4, ensure_ascii=False))