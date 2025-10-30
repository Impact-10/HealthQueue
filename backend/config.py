"""Shared configuration helpers for lightweight local inference.

This module centralizes environment-driven settings so that individual
model wrappers can remain lightweight and avoid accidental downloads.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

# Default modes for each model wrapper.  "fallback" means the wrapper will
# return a canned response or proxy implementation instead of loading the
# heavy model weights.  Switching a model to "inference" explicitly opts in
# to downloading/loading the underlying checkpoint.
DEFAULT_MODEL_MODES: Dict[str, str] = {
    "medalpaca": "inference",
    "biogpt": "inference",
    "longformer": "inference",
    "pubmedbert": "inference",
    "ensemble": "inference",
}

# Simple cache so repeated lookups avoid string parsing.
_MODEL_MODE_CACHE: Dict[str, str] = {}

# Guardrails limiting which checkpoints can be hydrated.  We explicitly
# restrict loading to small, community-friendly models and reject any env
# override that points at heavyweight 13B+ checkpoints.
ALLOWED_MODEL_REPOS: Dict[str, tuple[str, ...]] = {
    "medalpaca": ("medalpaca/medalpaca-7b",),
    "biogpt": ("microsoft/biogpt", "google/flan-t5-small"),
    "longformer": ("yikuan8/Clinical-Longformer",),
    "pubmedbert": ("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",),
}

FORBIDDEN_MODEL_KEYWORDS = ("13b", "33b", "34b", "40b", "65b", "70b", "175b")


def _normalise_key(model_key: str) -> str:
    return model_key.lower().strip()


def get_model_mode(model_key: str, default: Optional[str] = None) -> str:
    """Return the configured operating mode for a given model.

    Modes:
        - "fallback": do not load the heavy model, rely on canned logic
        - "inference": allow the wrapper to lazy-load the checkpoint

    The mode can be overridden via an environment variable named
    ``{MODEL_KEY}_MODE`` (upper-cased model key).
    """

    key = _normalise_key(model_key)
    if key in _MODEL_MODE_CACHE:
        return _MODEL_MODE_CACHE[key]

    env_var_name = f"{key.upper()}_MODE"
    env_value = os.getenv(env_var_name)

    if env_value:
        env_value = env_value.lower().strip()
        if env_value in {"fallback", "inference"}:
            _MODEL_MODE_CACHE[key] = env_value
            return env_value

    mode = DEFAULT_MODEL_MODES.get(key, default or "fallback")
    _MODEL_MODE_CACHE[key] = mode
    return mode


def is_inference_enabled(model_key: str, default: Optional[str] = None) -> bool:
    """Convenience helper to check if a model should load weights."""

    return get_model_mode(model_key, default) == "inference"


def get_timeout_seconds(model_key: str, default: int = 30) -> int:
    """Optional helper to read per-model timeout configuration."""

    env_var_name = f"{model_key.upper()}_TIMEOUT_SECONDS"
    value = os.getenv(env_var_name)
    if value and value.isdigit():
        return max(1, int(value))
    return default


def reset_model_mode_cache() -> None:
    """Clear cached values (useful in tests)."""

    _MODEL_MODE_CACHE.clear()


def ensure_allowed_model_name(model_key: str, model_name: str) -> str:
    """Validate that a configured checkpoint remains within policy."""

    if not model_name:
        raise ValueError(f"Missing model name for {model_key}.")

    candidate = model_name.strip()
    lower_candidate = candidate.lower()

    for keyword in FORBIDDEN_MODEL_KEYWORDS:
        if keyword in lower_candidate:
            raise ValueError(
                (
                    f"{model_key} model '{candidate}' contains forbidden keyword '{keyword}'. "
                    "Only lightweight (≤7B) checkpoints are permitted."
                )
            )

    allowlist = ALLOWED_MODEL_REPOS.get(_normalise_key(model_key))
    if allowlist and candidate not in allowlist:
        env_var = f"{model_key.upper()}_MODEL_NAME"
        allowed_list = ", ".join(sorted(allowlist))
        raise ValueError(
            (
                f"{env_var}={candidate} is not in the approved allowlist. "
                f"Choose one of: {allowed_list}."
            )
        )

    return candidate


@lru_cache(maxsize=1)
def get_model_cache_dir() -> Optional[str]:
    """Resolve and create the shared cache directory if configured."""

    cache_dir = os.getenv("MODEL_CACHE_DIR")
    if not cache_dir:
        return None

    path = Path(cache_dir).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # noqa: BLE001 - surface OS errors
        raise RuntimeError(f"Failed to create MODEL_CACHE_DIR '{path}': {exc}") from exc
    return str(path)


def with_cache_dir(kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Attach the configured Hugging Face cache directory to load kwargs."""

    resolved_kwargs: Dict[str, Any] = dict(kwargs or {})
    cache_dir = get_model_cache_dir()
    if cache_dir:
        resolved_kwargs.setdefault("cache_dir", cache_dir)
    return resolved_kwargs
