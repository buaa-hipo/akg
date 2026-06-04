import logging
from typing import Tuple, List

from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator.database.island import Island
from ai_kernel_generator.core.agent.utils.feature_extractor import FeatureExtractor
from ai_kernel_generator.core.agent.shared_resources import get_cross_encoder

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class CrossEncoderSimilarity:
    def __init__(self, model_name="/mnt/lustre-client/zhangzizheng/ALL_MODELS/models--cross-encoder--stsb-roberta-large/snapshots/2b12c2c0088918e76151fd5937b7bba986ef1f98"):
        """
        初始化Cross-Encoder模型
        :param model_name: 预训练模型名称或路径
        """
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        logger.info(f"Initialized Cross-Encoder model from {model_name} on {self.device}") 
    
    def calculate_similarity(self, text1, text2):
        """
        计算两个文本的相似度分数
        :param text1: 第一个文本
        :param text2: 第二个文本
        :return: 相似度分数(0-1之间)
        """
        # 准备模型输入
        inputs = self.tokenizer(text1, text2, truncation=True, return_tensors="pt").to(self.device)
        
        # 获取模型预测
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 将logits转换为相似度分数(0-1)
        score = torch.sigmoid(outputs.logits).item()
        
        return score

class Filter(AgentBase):
    def __init__(
        self,
        op_name: str,
        task_desc: str,
        dsl: str = "",
        backend: str = "",
        arch: str = "",
        island: Island = None,
        config: dict = None,
        cross_encoder=None,
    ):
        self.op_name = op_name
        self.task_desc = task_desc
        self.dsl = dsl
        self.arch = arch
        self.backend = backend
        self.island = island
        self.config = config

        # 从config中获取model_config
        if config:
            self.model_config = config.get("agent_model_config", {})
        else:
            raise ValueError("config is required for Designer")

        self.cross_encoder = cross_encoder or get_cross_encoder(config=config)

        context = {
            "agent_name": "filter",
            "dsl": self.dsl,
            "op_name": self.op_name,
            "backend": self.backend,
            "arch": self.arch,
            "task_desc": self.task_desc,
        }
        super().__init__(context=context, config=config)

    async def run(self, designer_ir: str) -> bool:
        # 根据 designer_ir 提取特征
        feature_extractor = FeatureExtractor(
            model_config=self.model_config,
            framework_code=self.task_desc,
            dsl=self.dsl,
            sketch_code=designer_ir
        )
        code_feat, _, _ = await feature_extractor.run()
        
        
        # 获取历史生成的代码 IR 和 特征
        exist_code_ir = self.island.get_exist_code_ir()
        exist_code_feat = self.island.get_exist_code_feat()
        
        # 使用CrossEncoder计算相似度，判断是否需要过滤
        # 计算 IR 和 特征 相似度分数
        import time
        start = time.time()
        ir_similarity_scores = [self.cross_encoder.calculate_similarity(designer_ir, exist_ir) for exist_ir in exist_code_ir]
        feat_similarity_scores = [self.cross_encoder.calculate_similarity(code_feat, exist_feat) for exist_feat in exist_code_feat]
        # 计算加权相似度分数，IR相似度权重为0.3，特征相似度权重为0.7
        weighted_similarity_scores = [0.3 * ir_score + 0.7 * feat_score for ir_score, feat_score in zip(ir_similarity_scores, feat_similarity_scores)]
        # 如果存在相似度分数超过0.85，则认为生成的代码与历史代码过于相似，需要过滤
        filter_or_not = any(score > 0.85 for score in weighted_similarity_scores)
        logger.info(f"Filter Score of current IR: {weighted_similarity_scores}")
        logger.info(f"Filter cross-encoder cost time: {time.time() - start:.2f}s")
        
        # DEBUG MODE
        import os
        if os.environ.get("AIKG_DEBUG_MODE", False):
            return False, code_feat
        
        return filter_or_not, code_feat
