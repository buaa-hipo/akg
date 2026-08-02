import logging
import os
import threading
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
        self.device = os.environ.get("AIKG_MODEL_DEVICE", "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._lock = threading.RLock()
        self.batch_size = max(1, int(os.environ.get("AIKG_CROSS_ENCODER_BATCH_SIZE", "16")))
        logger.info(f"Initialized Cross-Encoder model from {model_name} on {self.device}") 
    
    def calculate_similarity(self, text1, text2):
        """
        计算两个文本的相似度分数
        :param text1: 第一个文本
        :param text2: 第二个文本
        :return: 相似度分数(0-1之间)
        """
        scores = self.calculate_similarities([(text1, text2)])
        return scores[0] if scores else 0.0

    def calculate_similarities(self, text_pairs):
        if not text_pairs:
            return []

        scores = []
        with self._lock:
            for start in range(0, len(text_pairs), self.batch_size):
                batch_pairs = text_pairs[start:start + self.batch_size]
                text1_list = [text1 for text1, _ in batch_pairs]
                text2_list = [text2 for _, text2 in batch_pairs]
                inputs = self.tokenizer(
                    text1_list,
                    text2_list,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                scores.extend(torch.sigmoid(outputs.logits).reshape(-1).detach().cpu().tolist())

        return [float(score) for score in scores]

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
        ir_pairs = [(designer_ir, exist_ir) for exist_ir in exist_code_ir]
        feat_pairs = [(code_feat, exist_feat) for exist_feat in exist_code_feat]
        similarity_pairs = ir_pairs + feat_pairs

        if hasattr(self.cross_encoder, "calculate_similarities"):
            similarity_scores = self.cross_encoder.calculate_similarities(similarity_pairs)
            ir_similarity_scores = similarity_scores[:len(ir_pairs)]
            feat_similarity_scores = similarity_scores[len(ir_pairs):]
        else:
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
