# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import random
from typing import List, Dict
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from ai_kernel_generator.database.coder_vector_store import CoderVectorStore
from ai_kernel_generator.database.database import Database, RetrievalStrategy
from ai_kernel_generator import get_project_root
from ai_kernel_generator.utils.common_utils import get_md5_hash

logger = logging.getLogger(__name__)

# Path(get_project_root()).parent.parent /mnt/lustre-client/zhangzizheng/AIKG/akg/aikg/
DEFAULT_CODER_DATABASE_PATH = Path(get_project_root()).parent.parent / "triton_database" / "nvidia" / "TritonBench_G_v1"

class CoderDatabase(Database):
    # 单例模式实现
    _instances: Dict[str, 'CoderDatabase'] = {}
    _lock = False  # 简单的锁机制避免并发问题
    
    def __new__(cls, database_path: str = "", config: dict = None):
        database_path = database_path or str(DEFAULT_CODER_DATABASE_PATH)
        # 使用数据库路径作为实例的唯一标识
        instance_key = get_md5_hash(database_path=database_path)
        
        # 检查实例是否已存在
        if instance_key not in cls._instances or cls._instances[instance_key] is None:
            # 简单锁机制
            while cls._lock:
                pass
            cls._lock = True
            try:
                # 双重检查锁定模式
                if instance_key not in cls._instances or cls._instances[instance_key] is None:
                    cls._instances[instance_key] = super(CoderDatabase, cls).__new__(cls)
            finally:
                cls._lock = False
        
        return cls._instances[instance_key]
        
    def __init__(self, database_path: str = "", config: dict = None):
        # TODO __init__ 也要加锁防止多重初始化
        while self.__class__._lock:
            pass
        self.__class__._lock = True
        try:
            # 防止重复初始化
            if hasattr(self, '_initialized') and self._initialized:
                return
            self.database_path = database_path or str(DEFAULT_CODER_DATABASE_PATH)
            self.basic_vector_store = CoderVectorStore(
                database_path=self.database_path,
                embedding_model_name='/mnt/lustre-client/zhangzizheng/ALL_MODELS/Jerry0/text2vec-large-chinese',
                index_name="basic_vector_store",
                features=["basic"],
                config=config
            )
            self.schedule_vector_store = CoderVectorStore(
                database_path=self.database_path,
                embedding_model_name='/mnt/lustre-client/zhangzizheng/ALL_MODELS/Jerry0/text2vec-large-chinese',
                index_name="schedule_vector_store",
                features=["schedule"],
                config=config
            )
            self.memory_vector_store = CoderVectorStore(
                database_path=self.database_path,
                embedding_model_name='/mnt/lustre-client/zhangzizheng/ALL_MODELS/Jerry0/text2vec-large-chinese',
                index_name="memory_vector_store",
                features=["memory"],
                config=config
            )
            self.vector_stores = [self.basic_vector_store, self.schedule_vector_store, self.memory_vector_store]
            super().__init__(self.database_path, self.vector_stores, config)    
            
            self._initialized = True
        finally:
            self.__class__._lock = False
    
    def is_empty(self):
        capacity = 0
        for vs in self.vector_stores:
            capacity += vs.vector_store.index.ntotal
        return capacity == 0
    
    def hierarchy_search(self, features: dict, k: int = 5):
        """
        层次检索：
        1. 第一步：通过 basic_vector_store 检索基础特征，获取候选文档池（放大k值保证候选量）
        2. 第二步：基于候选池，分别通过 schedule_vector_store 和 memory_vector_store 做二次检索
        3. 第三步：融合二次检索结果，去重后取 top k 最终结果
        """
        # ---------------------- 第一步：基础特征检索（basic_vector_store） ----------------------
        # 构建basic特征的查询语句
        basic_query = ", ".join([f"{key}: {features[key]}" for key in self.basic_vector_store.features if key in features])
        # 基础检索取更多候选（保证后续二次检索有足够样本）
        basic_candidate_k = max(20, 5 * k)
        basic_docs = self.basic_vector_store.vector_store.similarity_search(
            query=basic_query,
            k=basic_candidate_k,
            # 可选：过滤相同算子类型和特征不变量的文档（根据业务需求调整）
            # filter={"feature_invariants": feature_invariantså}
        )

        if not basic_docs:
            return []

        # 提取基础候选文档的文件路径（用于后续二次检索的范围限定）
        basic_candidate_paths = {doc.metadata.get("file_path", "") for doc in basic_docs}
        if not basic_candidate_paths:
            raise ValueError("Basic candidate docs have no 'file_path' in metadata")

        # ---------------------- 第二步：调度/内存特征二次检索 ----------------------
        # 2.1 调度特征检索（schedule_vector_store）
        schedule_query = ", ".join([f"{key}: {features[key]}" for key in self.schedule_vector_store.features if key in features])
        schedule_docs = self.schedule_vector_store.vector_store.similarity_search(
            query=schedule_query,
            k=basic_candidate_k,
            # filter={"feature_invariants": feature_invariants, "op_type": op_type}
        )
        # 过滤出在基础候选池中的调度相关文档
        schedule_filtered_docs = [doc for doc in schedule_docs if doc.metadata.get("file_path") in basic_candidate_paths]

        # 2.2 内存特征检索（memory_vector_store）
        memory_query = ", ".join([f"{key}: {features[key]}" for key in self.memory_vector_store.features if key in features])
        memory_docs = self.memory_vector_store.vector_store.similarity_search(
            query=memory_query,
            k=basic_candidate_k,
            # filter={"feature_invariants": feature_invariants, "op_type": op_type}
        )
        # 过滤出在基础候选池中的内存相关文档
        memory_filtered_docs = [doc for doc in memory_docs if doc.metadata.get("file_path") in basic_candidate_paths]

        # ---------------------- 第三步：结果融合与去重 ----------------------
        # 合并schedule和memory的检索结果
        merged_docs = schedule_filtered_docs + memory_filtered_docs
        if not merged_docs:
            # 若二次检索无结果，降级使用基础检索的top k
            final_docs = basic_docs[:k]
        else:
            # 去重（按file_path）
            seen_paths = set()
            unique_docs = []
            for doc in merged_docs:
                path = doc.metadata.get("file_path")
                if path not in seen_paths:
                    seen_paths.add(path)
                    unique_docs.append(doc)
            
            # 取top k，若不足则补充基础检索的结果
            # 随机选取 k 个，若不足则补充基础检索结果
            random.shuffle(unique_docs)
            final_docs = unique_docs[:k]
            if len(final_docs) < k:
                # 补充基础检索中未被选中的文档
                basic_supplement = [doc for doc in basic_docs if doc.metadata.get("file_path") not in seen_paths]
                final_docs += basic_supplement[:k - len(final_docs)]

        return final_docs


    async def samples(self, output_content:List[str], sample_num:int = 1, code_feat: str="", impl_code: str = "", framework_code:str = "",
                      backend: str = "", arch: str = "", dsl: str = "", framework: str = "", sketch_code: str = ""):
        """
        Evolve采样方案，根据当前算子的特征信息，从数据库中采样出优化性和随机性的算子实现。
        """
        need_extract_features = False
        for vector_store in self.vector_stores:
            if vector_store.enable_vector_store:
                need_extract_features = True
                break
        
        if need_extract_features:
            code_feat = await self.extract_features(code_feat, impl_code, framework_code, backend, arch, dsl, sketch_code)
            # feature_invariants = get_md5_hash(backend=backend, arch=arch, dsl=dsl)
            
            docs = self.hierarchy_search(code_feat, sample_num)
            result = self.get_output_content(output_content, RetrievalStrategy.HIERARCHY, docs, dsl, framework)
        else:
            random_res = self.randomicity_search(output_content, sample_num, backend, arch, dsl, framework)
            result = random_res
        
        return result
