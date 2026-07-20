import json
from pathlib import Path
import argparse


def _is_non_empty_id(value):
    return isinstance(value, str) and value.strip()


def _node_to_trace_item(node, edge_from_prev=None):
    profile = node.get("profile", {})
    return {
        "id": node.get("id"),
        "parent_id": node.get("parent_id") or "",
        "fallback_id": node.get("fallback_id") or "",
        "edge_from_prev": edge_from_prev,
        "early_stopping_reason": node.get("early_stopping_reason", ""),
        "round": node.get("round"),
        "task_id": node.get("task_id", ""),
        "unique_dir": node.get("unique_dir", ""),
        "speedup": profile.get("speedup", float("-inf")),
    }


def _append_node(path_stack, node, edge_from_prev):
    if node is None:
        return
    path_stack.append(_node_to_trace_item(node, edge_from_prev=edge_from_prev))


def get_evolution_trajectories(base_path):
    all_nodes = {}
    referenced_ids = set()

    # 1. 加载所有节点信息
    path = Path(base_path)
    for json_file in path.glob("*/impl_info.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                node_id = data.get('id')
                if node_id:
                    all_nodes[node_id] = data
                    
                    # 只用 parent_id 判断父子树叶子。fallback_id 表示触发早停的旧分支，
                    # 不应让旧分支节点失去叶子身份，否则 B->C 这种停止分支会被隐藏。
                    p_id = data.get('parent_id')
                    if _is_non_empty_id(p_id):
                        referenced_ids.add(p_id)
        except Exception as e:
            print(f"读取文件 {json_file} 出错: {e}")

    # 2. 识别叶子节点 (没有出现在任何节点的 parent_id 引用中)
    leaf_ids = [node_id for node_id in all_nodes if node_id not in referenced_ids]

    trajectories = []

    # 3. 对每个叶子节点沿 parent_id 回溯。path_stack 是叶子到根方向，最后再反转。
    #    如果当前节点存在 fallback_id，表示 parent_id -> current_id 这条边来自回退，
    #    正向 trace 中表现为 parent_id => (fallback_id)current_id。
    for leaf_id in leaf_ids:
        path_stack = []
        curr_id = leaf_id
        seen = set()
        
        while curr_id in all_nodes:
            if curr_id in seen:
                print(f"检测到环路，停止回溯: {curr_id}")
                break
            seen.add(curr_id)

            node = all_nodes[curr_id]
            fallback_id = node.get("fallback_id")
            parent_id = node.get("parent_id")
            
            if _is_non_empty_id(parent_id):
                edge_from_prev = "fallback" if _is_non_empty_id(fallback_id) else "parent"
                _append_node(path_stack, node, edge_from_prev=edge_from_prev)
                curr_id = parent_id
            else:
                _append_node(path_stack, node, edge_from_prev=None)
                break  # 到达根节点

        # 4. 格式化：反转顺序并修正首节点边类型
        path_stack.reverse()
        if path_stack:
            path_stack[0]["edge_from_prev"] = None
        for depth, node_info in enumerate(path_stack):
            node_info["depth"] = depth
            
        trajectories.append(path_stack)

    return trajectories

# def export_folder(folder_path: str):
    

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="传入文件夹路径参数")
    
    # 2. 添加 --folder_path 参数（必填）
    parser.add_argument(
        "--folder_path", 
        type=str, 
        required=True, 
        default="/mnt/lustre-client/zhangzizheng/AIKG/akg/aikg/evolve_database/level2/59_Matmul_Swish_Scaling/island_0",
        help="请输入你的实际文件夹路径"
    )
    
    # 3. 解析命令行参数
    args = parser.parse_args()
    
    # 4. 获取传入的路径
    folder_path = args.folder_path
    
    for old_file in Path(".").glob("evolve_*.json"):
        old_file.unlink()
    
    results = get_evolution_trajectories(folder_path)
    results.sort(key=lambda x: len(x), reverse=True)
    print(f"一共有 {len(results)} 条进化轨迹 (叶子节点)")
    for track in results:
        for idx, t in enumerate(track):
            if idx > 0:
                if t.get("edge_from_prev") == "fallback":
                    fallback_id = t.get("fallback_id", "")
                    fallback_label = f"(id{fallback_id[:3]})" if _is_non_empty_id(fallback_id) else ""
                    print(f" => {fallback_label}", end="")
                else:
                    print(" -> ", end="")
            print(f"id{t['id'][:3]}_{t['speedup']:.2f}x", end="")
        print()
    
    # 将结果转换为JSON字符串输出
    for idx, result in enumerate(results):
        # 导出到json文件
        with open(f"evolve_{idx}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        
    # print(json.dumps(results, indent=4, ensure_ascii=False))
