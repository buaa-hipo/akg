import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

_RESOURCE_LOCK = threading.Lock()
_RESOURCE_PID: Optional[int] = None
_CROSS_ENCODER: Any = None
_CODER_DATABASE: Any = None

_WORKFLOW_REQUIRED_AGENTS = {
    "default": {"designer", "coder", "verifier"},
    "default_workflow": {"designer", "coder", "verifier"},
    "coder_only": {"coder", "verifier"},
    "coder_only_workflow": {"coder", "verifier"},
    "verifier_only": {"verifier"},
    "verifier_only_workflow": {"verifier"},
    "connect_all": {"designer", "coder", "verifier"},
    "conductor_connect_all_workflow": {"designer", "coder", "verifier"},
    "designer_only": {"designer", "filter"},
    "designer_only_workflow": {"designer", "filter"},
    "evolve_workflow": {"designer", "filter", "coder", "verifier"},
}


def _normalize_workflow_name(workflow_name: Optional[str]) -> str:
    if not workflow_name or workflow_name == "default":
        return "default_workflow"
    return workflow_name


def get_required_agents_for_workflow(workflow_name: Optional[str]) -> set[str]:
    normalized_name = _normalize_workflow_name(workflow_name)
    return _WORKFLOW_REQUIRED_AGENTS.get(normalized_name, {"designer", "coder", "verifier"})


def _reset_cache_if_pid_changed_locked() -> None:
    global _RESOURCE_PID, _CROSS_ENCODER, _CODER_DATABASE

    current_pid = os.getpid()
    if _RESOURCE_PID == current_pid:
        return

    if _RESOURCE_PID is not None:
        logger.info(
            "Detected process change, resetting shared resources: old pid=%s, new pid=%s",
            _RESOURCE_PID,
            current_pid,
        )
    _RESOURCE_PID = current_pid
    _CROSS_ENCODER = None
    _CODER_DATABASE = None


def _get_nested_config(config: Optional[dict], section: str, key: str) -> Optional[Any]:
    section_config = (config or {}).get(section, {})
    if isinstance(section_config, dict):
        return section_config.get(key)
    return None


def _get_service_url(config: Optional[dict], kind: str) -> str:
    service_config = (config or {}).get("service_config", {})
    database_config = (config or {}).get("database_config", {})
    service_config = service_config if isinstance(service_config, dict) else {}
    database_config = database_config if isinstance(database_config, dict) else {}

    if kind == "cross_encoder":
        candidates = [
            service_config.get("cross_encoder_url"),
            service_config.get("model_service_url"),
            os.getenv("AIKG_CROSS_ENCODER_SERVICE_URL"),
            os.getenv("AIKG_MODEL_SERVICE_URL"),
        ]
    elif kind == "coder_database":
        candidates = [
            database_config.get("service_url"),
            database_config.get("model_service_url"),
            service_config.get("coder_database_url"),
            service_config.get("model_service_url"),
            os.getenv("AIKG_CODER_DATABASE_SERVICE_URL"),
            os.getenv("AIKG_MODEL_SERVICE_URL"),
        ]
    else:
        candidates = []

    for value in candidates:
        if value:
            return str(value).rstrip("/")
    return ""


def get_cross_encoder(config: Optional[dict] = None):
    global _CROSS_ENCODER

    with _RESOURCE_LOCK:
        _reset_cache_if_pid_changed_locked()
        if _CROSS_ENCODER is None:
            service_url = _get_service_url(config, "cross_encoder")
            if service_url:
                from ai_kernel_generator.client.model_service_client import RemoteCrossEncoderSimilarity

                _CROSS_ENCODER = RemoteCrossEncoderSimilarity(service_url)
                logger.info("Remote CrossEncoderSimilarity initialized for pid=%s", _RESOURCE_PID)
            else:
                # local import to avoid circular import
                from ai_kernel_generator.core.agent.filter import CrossEncoderSimilarity

                _CROSS_ENCODER = CrossEncoderSimilarity()
                logger.info("Shared CrossEncoderSimilarity initialized for pid=%s", _RESOURCE_PID)
        return _CROSS_ENCODER


def get_coder_database(config: Optional[dict] = None):
    global _CODER_DATABASE

    with _RESOURCE_LOCK:
        _reset_cache_if_pid_changed_locked()
        if _CODER_DATABASE is None:
            service_url = _get_service_url(config, "coder_database")
            if service_url:
                from ai_kernel_generator.client.model_service_client import RemoteCoderDatabase

                database_path = _get_nested_config(config, "database_config", "database_path") or ""
                _CODER_DATABASE = RemoteCoderDatabase(service_url, config=config, database_path=database_path)
                logger.info("Remote CoderDatabase initialized for pid=%s", _RESOURCE_PID)
            else:
                from ai_kernel_generator.database.coder_database import CoderDatabase

                _CODER_DATABASE = CoderDatabase(config=config)
                logger.info("Shared CoderDatabase initialized for pid=%s", _RESOURCE_PID)
        return _CODER_DATABASE


def warmup_for_workflow(workflow_name: Optional[str], config: Optional[dict] = None) -> None:
    required_agents = get_required_agents_for_workflow(workflow_name)

    if "filter" in required_agents:
        get_cross_encoder(config=config)

    if "coder" in required_agents:
        database_config = (config or {}).get("database_config", {})
        if database_config.get("enable_rag", False):
            get_coder_database(config=config)
