import logging
import asyncio
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

logger = logging.getLogger(__name__)


class RemoteCrossEncoderSimilarity:
    """HTTP client with the same public methods as CrossEncoderSimilarity."""

    def __init__(self, service_url: str, timeout: float = 120.0):
        self.service_url = service_url.rstrip("/")
        self.timeout = timeout
        logger.info("Using remote CrossEncoderSimilarity service: %s", self.service_url)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        scores = self.calculate_similarities([(text1, text2)])
        return scores[0] if scores else 0.0

    def calculate_similarities(self, text_pairs: Sequence[Tuple[str, str]]) -> List[float]:
        if not text_pairs:
            return []

        payload = {
            "pairs": [{"text1": text1, "text2": text2} for text1, text2 in text_pairs]
        }
        response = requests.post(
            f"{self.service_url}/api/v1/similarity/batch",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        scores = [float(score) for score in data.get("scores", [])]
        if len(scores) != len(text_pairs):
            raise RuntimeError(
                f"Remote similarity service returned {len(scores)} scores for {len(text_pairs)} pairs"
            )
        return scores


class RemoteCoderDatabase:
    """HTTP client exposing the subset of CoderDatabase used by Coder."""

    def __init__(
        self,
        service_url: str,
        config: Optional[dict] = None,
        database_path: str = "",
        timeout: float = 300.0,
    ):
        self.service_url = service_url.rstrip("/")
        self.config = {"agent_model_config": (config or {}).get("agent_model_config", {})}
        self.database_path = str(database_path or "")
        self.timeout = timeout
        logger.info("Using remote CoderDatabase service: %s", self.service_url)

    async def samples(
        self,
        output_content: List[str],
        sample_num: int = 1,
        code_feat: str = "",
        impl_code: str = "",
        framework_code: str = "",
        backend: str = "",
        arch: str = "",
        dsl: str = "",
        framework: str = "",
        sketch_code: str = "",
    ) -> List[Dict[str, Any]]:
        payload = {
            "output_content": output_content,
            "sample_num": sample_num,
            "code_feat": code_feat,
            "impl_code": impl_code,
            "framework_code": framework_code,
            "backend": backend,
            "arch": arch,
            "dsl": dsl,
            "framework": framework,
            "sketch_code": sketch_code,
            "database_path": self.database_path,
            "config": self.config,
        }
        data = await asyncio.to_thread(self._post_json, "/api/v1/coder_database/samples", payload)
        return data.get("samples", [])

    async def is_empty(self) -> bool:
        params = {}
        if self.database_path:
            params["database_path"] = self.database_path
        data = await asyncio.to_thread(self._get_json, "/api/v1/coder_database/is_empty", params)
        return bool(data.get("is_empty", True))

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{self.service_url}{path}",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _get_json(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.get(
            f"{self.service_url}{path}",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
