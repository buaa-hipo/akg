from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from typing import List, Dict, Any, Optional
from enum import Enum, auto
from ai_kernel_generator.database.embed import embed_single, embed_py2vecs
import uuid
from datetime import datetime
from sentence_transformers import SentenceTransformer

class CollectorState(Enum):
    """PointCollector 的状态机枚举。

    语义约定：
    - START:        当前不存在待配对的失败（还未失败，或上一轮 episode 已结束）。
    - PENDING_FAIL: 已记录一次失败（上一轮为 False），等待下一次结果组成 episode。
    - SUCCEED:      当前不存在待配对的失败，且最近一次结果为成功。
    """

    START = auto()
    PENDING_FAIL = auto()
    SUCCEED = auto()


class PointCollector:
    """负责将错误场景及相关代码收集为 Qdrant 向量的收集器。

    该类被设计为一个简单状态机，具有三个状态：
    - EMPTY:        未收集任何数据
    - ERROR_AND_OLD: 已收集错误描述和 old_code
    - ERROR_OLD_NEW: 已收集错误描述、old_code 和 new_code
    """

    def __init__(self, client: QdrantClient, collection_name: str, task_info: dict):
        self.client = client
        self.collection_name = collection_name
        self.points: List[models.PointStruct] = []
        self.task_info = task_info
        
        self.encoder = SentenceTransformer(
            "microsoft/unixcoder-base",
            cache_folder="/mnt/lustre-client/lutao/huggingface"
        )

        # 状态相关字段
        self.state: CollectorState = CollectorState.START
        self.error_description: Optional[str] = None
        self.error_context: Optional[Dict[str, Any]] = None
        self.old_code: Optional[str] = None
        # new_code 与 repair_code 等价，保留 repair_code 以兼容后续可能的使用
        self.new_code: Optional[str] = None
        self.repair_code: Optional[str] = None
        # 是否有效性验证列表
        self.varify_list = []

        # 最近一次失败和最近一次 episode 记录
        self.last_episode_type: Optional[str] = None  # "fail_success" 或 "fail_fail"
        self.last_episode_first_code: Optional[str] = None
        self.last_episode_first_error: Optional[str] = None
        self.last_episode_second_code: Optional[str] = None
        self.last_episode_second_error: Optional[str] = None

    def insert_error_point(
        self,
        code_contexts: List[str],
        error_description: str,
        payload_extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        插入一个 Point：
        - named_vec: error_description 的向量
        - named_multivec: 所有 code_paths 对应代码片段的 multivec
        - payload: 包含 code_paths / error_description 以及额外字段
        - error_description: 错误描述文本, 用于初筛
        """
        # 1) 单向量：错误描述
        err_vec = embed_single(error_description, self.encoder)           # shape=(dim,)

        # 2) multivec：多个文件的所有语句
        code_multivec = []
        for code in code_contexts:
            # m = self.encoder._first_module().auto_model
            # print("max_position_embeddings:", m.config.max_position_embeddings)
            # tok = self.encoder.tokenizer
            # ids = tok(code_contexts[0], truncation=False, return_tensors="pt")["input_ids"]
            # print("seq_len:", ids.shape[1], "max_token_id:", int(ids.max()))
            
            code_multivec += embed_py2vecs(code, self.encoder)  # shape=(N, dim)

        # 3) named_vec / named_multivec 结构
        vectors = {
            "error_desc_vec": err_vec,
            "code_multivec": code_multivec
        }

        # 4) payload
        payload: Dict[str, Any] = {
            "error_description": error_description,
            "code_contexts": code_contexts,
        }
        if payload_extra:
            payload.update(payload_extra)

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vectors,
                    payload=payload,
                )
            ],
        )

    def record_run(self, verify_res: bool, code: str, error_stack: Optional[str]) -> None:
        """记录一次验证结果，并根据两次连续运行构造 episode。

        规则：
        - EMPTY 状态 + True: 不形成 episode，保持 EMPTY。
        - EMPTY 状态 + False: 进入 ERROR_AND_OLD，记为第一条失败记录。
        - ERROR_AND_OLD + True: 形成一次 fail_success episode，然后重置为 EMPTY。
        - ERROR_AND_OLD + False: 形成一次 fail_fail episode，然后当前失败变成新的 ERROR_AND_OLD。
        """

        self.varify_list.append(verify_res)

        if self.state is CollectorState.START:
            if verify_res:
                # 首次或上一 episode 结束后直接成功，进入 SUCCEED
                self.state = CollectorState.SUCCEED
                return
            # 首次失败，进入挂起失败状态
            self.error_description = error_stack
            self.old_code = code
            self.new_code = None
            self.repair_code = None
            self.state = CollectorState.PENDING_FAIL

        elif self.state is CollectorState.SUCCEED:
            # CollectorState.SUCCEED 是终止状态
            return

        elif self.state is CollectorState.PENDING_FAIL:
            if verify_res:
                # fail -> success
                self.last_episode_type = "fail_success"
                self.last_episode_first_code = self.old_code
                self.last_episode_first_error = self.error_description
                self.last_episode_second_code = code
                self.last_episode_second_error = None
                # insert point to database
                payload={
                    "episode_type": self.last_episode_type,
                    "error_code": self.last_episode_first_code,
                    "error_description": self.last_episode_first_error,
                    "repair_code": self.last_episode_second_code,
                    "timestamp": datetime.now().strftime("%Y/%m/%d/%H/%M/%S")
                }
                for k in ("op_name", "task_id", "dsl", "backend", "arch", "framework", "workflow_name"):
                    v = self.task_info.get(k)
                    if v:
                        payload[k] = v

                try:     
                    self.insert_error_point(
                        code_contexts=[self.old_code],          # only old_code is the key
                        error_description=self.error_description,
                        payload_extra=payload
                    )
                except Exception as e:
                    print("[PointCollector] Qdrant upsert error:", repr(e))                

                # 本轮 episode 完结，清空挂起失败，进入 SUCCEED
                self.error_description = None
                self.error_context = None
                self.old_code = None
                self.new_code = None
                self.repair_code = None
                self.state = CollectorState.SUCCEED
            else:
                # fail -> fail
                self.last_episode_type = "fail_fail"
                self.last_episode_first_code = self.old_code
                self.last_episode_first_error = self.error_description
                self.last_episode_second_code = code
                self.last_episode_second_error = error_stack
                # insert point to database
                payload={
                    "episode_type": self.last_episode_type,
                    "error_code": self.last_episode_first_code,
                    "error_description": self.last_episode_first_error,
                    "repair_code": self.last_episode_second_code,
                    "repair_error_description": self.last_episode_second_error,
                    "timestamp": datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                }
                for k in ("op_name", "task_id", "dsl", "backend", "arch", "framework", "workflow_name"):
                    v = self.task_info.get(k)
                    if v:
                        payload[k] = v
                try:
                    self.insert_error_point(
                        code_contexts=[self.old_code],          # only old_code is the key
                        error_description=self.error_description,
                        payload_extra=payload
                    )
                except Exception as e:
                    print("[PointCollector] Qdrant upsert error:", repr(e))

                # 将当前失败作为新的挂起失败，继续等待下一次
                self.error_description = error_stack
                self.old_code = code
                self.new_code = None
                self.repair_code = None
                self.state = CollectorState.PENDING_FAIL
    
    # ==================== 状态查询接口 ====================
    def get_state(self) -> CollectorState:
        """返回当前状态枚举。"""

        return self.state

    def is_empty(self) -> bool:
        """是否处于 START 状态（不存在待配对的失败）。"""

        return self.state is CollectorState.START

    def has_error_and_old(self) -> bool:
        """是否已收集错误描述和 old_code。"""

        return self.state is CollectorState.PENDING_FAIL

    def has_error_old_and_new(self) -> bool:
        """是否已收集错误描述、old_code 和 new_code。"""

        return self.state is CollectorState.SUCCEED

    # ==================== 状态转换接口 ====================
    def reset(self) -> None:
        """重置为 EMPTY 状态并清空已收集数据。"""

        self.state = CollectorState.START
        self.error_description = None
        self.error_context = None
        self.old_code = None
        self.new_code = None
        self.repair_code = None


if __name__ == "__main__":
    client = QdrantClient("http://172.17.0.1:6333")  
    collection_name = "lxc_error_cases"
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            "error_desc_vec": models.VectorParams(
                size=768,
                distance=models.Distance.COSINE,
            ),
            "code_multivec": models.VectorParams(
                size=768,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        },
        on_disk_payload=True,
    )
