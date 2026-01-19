from sentence_transformers import SentenceTransformer
import numpy as np
import ast
from typing import List, Dict, Any, Optional

def split_python_file_into_chunks(code: str):
    lines = code.split("\n")
    tree = ast.parse(code)

    chunks = []

    def extract_func_header(child):
        """
        恢复函数定义（包括装饰器 + 多行参数）
        返回：
            - full_def
            - signature
            - decorators
            - header_end_line_idx（header 在文件中的结束行索引，0-based）
        """
        def_start = child.lineno - 1

        # 1) 向上找装饰器
        decor_start = def_start
        while decor_start - 1 >= 0 and lines[decor_start - 1].lstrip().startswith("@"):
            decor_start -= 1

        decorators = [lines[i].strip() for i in range(decor_start, def_start)]

        # 2) 找完整函数头（多行参数直到 ':'）
        header_lines = []
        header_end = def_start
        for i in range(def_start, len(lines)):
            header_lines.append(lines[i])
            header_end = i
            if lines[i].rstrip().endswith(":"):
                break

        signature = "\n".join(header_lines).rstrip()
        full_def_lines = lines[decor_start:header_end + 1]
        full_def = "\n".join(full_def_lines).rstrip()

        return full_def, signature, decorators, header_end

    def node_end_lineno(node):
        """获取节点的结束行号（1-based），兼容没有 end_lineno 的情况。"""
        end = getattr(node, "end_lineno", None)
        if end is not None:
            return end
        # 回退：遍历子节点取最大 lineno
        max_line = node.lineno
        for sub in ast.walk(node):
            if hasattr(sub, "lineno"):
                max_line = max(max_line, sub.lineno)
        return max_line

    def extract_chunk(node, parent_class=None):
        if not hasattr(node, "body"):
            return

        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                symbol = child.name
                if parent_class:
                    symbol = f"{parent_class}.{symbol}"

                start = child.lineno - 1
                end = node_end_lineno(child)      # 1-based
                # 完整代码（包括定义 + 函数体）
                code_text = "\n".join(lines[start:end])

                # 提取完整定义，并获取 header 的结束行号（0-based）
                full_def, signature, decorators, header_end = extract_func_header(child)

                # ===== 按语句切分函数体 =====
                stmt_chunks = []
                for stmt in child.body:
                    s_start = stmt.lineno - 1               # 0-based
                    s_end = node_end_lineno(stmt)           # 1-based
                    stmt_text = "\n".join(lines[s_start:s_end])
                    stmt_chunks.append({
                        "stmt_code": stmt_text,
                        "stmt_start_line": s_start + 1,
                        "stmt_end_line": s_end,
                        "stmt_type": type(stmt).__name__,
                    })

                chunks.append({
                    "symbol": symbol,
                    "start_line": start + 1,
                    "end_line": end,
                    "full_def": full_def,
                    "signature": signature,
                    "decorators": decorators,
                    "code": code_text,
                    # 每个元素是一个“语句级”块，可能跨多行
                    "code_stmts": stmt_chunks,
                })

            elif isinstance(child, ast.ClassDef):
                extract_chunk(child, parent_class=child.name)

    extract_chunk(tree)
    return chunks

def _tail_lines(text: Optional[str], n: int = 10) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-n:])

def embed_single(texts, encoder: SentenceTransformer):
    texts = _tail_lines(texts, n=5)
    return encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)


def embed_multivector(text: str, encoder: SentenceTransformer) -> np.ndarray:
    """
    输入：一段代码字符串
    输出：形状 [num_tokens, dim] 的 ndarray，可以直接当作 multivector
    """
    token_emb = encoder.encode(
        text,
        output_value="token_embeddings",
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if token_emb.ndim == 3:
        token_emb = token_emb[0]
    return token_emb  # shape: (num_tokens, 768)


def embed_py2vecs(code: str, encoder: SentenceTransformer):
    """
    输入：triton 文件内容
    输出：每个函数语句块的嵌入向量
    """
    funcs = split_python_file_into_chunks(code)
    res_list = []
    for func in funcs:
        for stmt in func["code_stmts"]:
            to_embed = func["full_def"] + "\n" + stmt["stmt_type"] + " " + stmt["stmt_code"]
            res_list.append(embed_single(to_embed, encoder))
    return res_list
