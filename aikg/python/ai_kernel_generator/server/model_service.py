import argparse
import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ai_kernel_generator import get_project_root
from ai_kernel_generator.core.agent.filter import CrossEncoderSimilarity
from ai_kernel_generator.database.coder_database import CoderDatabase
from ai_kernel_generator.utils.common_utils import load_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AIKG Model Service")

_SERVICE_CONFIG: Optional[dict] = None
_SERVICE_CONFIG_PATH = os.getenv("AIKG_MODEL_SERVICE_CONFIG", "")
_DATABASE_PATH = os.getenv("AIKG_CODER_DATABASE_PATH", "")
_CROSS_ENCODER: Optional[CrossEncoderSimilarity] = None
_CODER_DATABASES: Dict[str, CoderDatabase] = {}
_CROSS_ENCODER_LOCK = threading.Lock()
_CODER_DATABASE_LOCK = threading.Lock()
_CODER_DATABASE_SEMAPHORE = asyncio.Semaphore(
    max(1, int(os.getenv("AIKG_CODER_DATABASE_MAX_CONCURRENCY", "1")))
)


class TextPair(BaseModel):
    text1: str
    text2: str


class SimilarityBatchRequest(BaseModel):
    pairs: List[TextPair]


class CoderSamplesRequest(BaseModel):
    output_content: List[str]
    sample_num: int = 1
    code_feat: str = ""
    impl_code: str = ""
    framework_code: str = ""
    backend: str = ""
    arch: str = ""
    dsl: str = ""
    framework: str = ""
    sketch_code: str = ""
    database_path: str = ""
    config: Optional[Dict[str, Any]] = None


def _resolve_path(path: str) -> Path:
    raw_path = Path(os.path.expanduser(path))
    if raw_path.is_absolute():
        return raw_path
    return Path(get_project_root()) / raw_path


def _load_config_from_path(config_path: str) -> dict:
    resolved_path = _resolve_path(config_path)
    logger.info("Loading model service config: %s", resolved_path)
    return load_yaml(str(resolved_path))


def _get_config(fallback: Optional[dict] = None) -> dict:
    global _SERVICE_CONFIG

    if _SERVICE_CONFIG is not None:
        return _SERVICE_CONFIG
    if _SERVICE_CONFIG_PATH:
        _SERVICE_CONFIG = _load_config_from_path(_SERVICE_CONFIG_PATH)
        return _SERVICE_CONFIG
    if fallback:
        return fallback
    return {"agent_model_config": {"feature_extractor": "deepseek_r1_default"}}


def _get_cross_encoder() -> CrossEncoderSimilarity:
    global _CROSS_ENCODER

    if _CROSS_ENCODER is None:
        with _CROSS_ENCODER_LOCK:
            if _CROSS_ENCODER is None:
                _CROSS_ENCODER = CrossEncoderSimilarity()
    return _CROSS_ENCODER


def _get_coder_database(database_path: str = "", config: Optional[dict] = None) -> CoderDatabase:
    resolved_database_path = database_path or _DATABASE_PATH
    database_key = str(Path(resolved_database_path).expanduser()) if resolved_database_path else "__default__"
    resolved_database_path = database_key if resolved_database_path else ""

    if database_key not in _CODER_DATABASES:
        with _CODER_DATABASE_LOCK:
            if database_key not in _CODER_DATABASES:
                _CODER_DATABASES[database_key] = CoderDatabase(
                    database_path=resolved_database_path,
                    config=_get_config(config),
                )
    return _CODER_DATABASES[database_key]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/similarity/batch")
async def calculate_similarity_batch(req: SimilarityBatchRequest):
    if not req.pairs:
        return {"scores": []}

    try:
        model = _get_cross_encoder()
        pairs = [(pair.text1, pair.text2) for pair in req.pairs]
        scores = await asyncio.to_thread(model.calculate_similarities, pairs)
        return {"scores": scores}
    except Exception as exc:
        logger.exception("Similarity request failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/coder_database/samples")
async def coder_database_samples(req: CoderSamplesRequest):
    try:
        database = _get_coder_database(req.database_path, req.config)
        async with _CODER_DATABASE_SEMAPHORE:
            samples = await database.samples(
                output_content=req.output_content,
                sample_num=req.sample_num,
                code_feat=req.code_feat,
                impl_code=req.impl_code,
                framework_code=req.framework_code,
                backend=req.backend,
                arch=req.arch,
                dsl=req.dsl,
                framework=req.framework,
                sketch_code=req.sketch_code,
            )
        return {"samples": samples}
    except Exception as exc:
        logger.exception("Coder database samples request failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v1/coder_database/is_empty")
async def coder_database_is_empty(database_path: str = ""):
    try:
        database = _get_coder_database(database_path)
        return {"is_empty": database.is_empty()}
    except Exception as exc:
        logger.exception("Coder database status request failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/warmup")
async def warmup(load_cross_encoder: bool = True, load_coder_database: bool = True):
    try:
        if load_cross_encoder:
            await asyncio.to_thread(_get_cross_encoder)
        if load_coder_database:
            await asyncio.to_thread(_get_coder_database)
        return {"status": "warmed"}
    except Exception as exc:
        logger.exception("Warmup failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def start_model_service(
    host: str = "0.0.0.0",
    port: int = 8010,
    config_path: str = "",
    database_path: str = "",
):
    global _SERVICE_CONFIG, _SERVICE_CONFIG_PATH, _DATABASE_PATH

    env_config_path = os.getenv("AIKG_MODEL_SERVICE_CONFIG", "")
    env_database_path = os.getenv("AIKG_CODER_DATABASE_PATH", "")
    config_path = config_path or env_config_path
    _SERVICE_CONFIG_PATH = config_path
    _DATABASE_PATH = database_path or env_database_path

    if config_path:
        _SERVICE_CONFIG = _load_config_from_path(config_path)

    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="Start AIKG model/retrieval service")
    parser.add_argument("--host", default=os.getenv("AIKG_MODEL_SERVICE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("AIKG_MODEL_SERVICE_PORT", "8010")))
    parser.add_argument("--config-path", default="")
    parser.add_argument("--database-path", default="")
    args = parser.parse_args()

    start_model_service(
        host=args.host,
        port=args.port,
        config_path=args.config_path,
        database_path=args.database_path,
    )


if __name__ == "__main__":
    main()
