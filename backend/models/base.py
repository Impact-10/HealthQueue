"""Base classes and utilities for lightweight medical model wrappers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import time

import torch

from config import get_model_mode, is_inference_enabled


class BaseMedicalModel(ABC):
    """Common functionality shared by all medical model backends.

    The goal is to keep every model wrapper lightweight by:
      * Deferring heavy downloads until explicitly requested via env vars
      * Offering a consistent fallback pathway with structured responses
      * Capturing minimal telemetry about load attempts and failures
    """

    fallback_label = "fallback"
    inference_label = "inference"

    def __init__(
        self,
        model_key: str,
        *,
        default_mode: str = fallback_label,
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        self.model_key = model_key
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._is_initialized = False
        self._last_error: Optional[str] = None
        self._metrics: Dict[str, Any] = {
            "load_attempts": 0,
            "successful_loads": 0,
            "last_loaded_at": None,
        }

        # Device selection is centralized here so each model can simply refer to
    # 'self.device'. Models that must stay on CPU can override the value
    # after calling 'super().__init__'.
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine the initial operating mode based on environment variables.
        mode = get_model_mode(model_key, default_mode)
        self._fallback_mode = mode != self.inference_label

    # ------------------------------------------------------------------
    # Core lifecycle helpers
    # ------------------------------------------------------------------
    def use_fallback(self) -> bool:
        """Return whether the model should return canned responses."""

        return self._fallback_mode

    def enable_fallback(self, reason: Optional[Exception] = None) -> None:
        """Switch to fallback mode and record the failure reason."""

        self._fallback_mode = True
        if reason is not None:
            self._last_error = str(reason)

    # Lightweight accessor so callers can surface meaningful errors when
    # inference was explicitly requested but we could not hydrate weights.
    def require_inference_disabled(self) -> None:
        """Raise when inference was requested but only fallback is available."""

        if is_inference_enabled(self.model_key):
            env_hint = f"Set {self.model_key.upper()}_MODE=fallback to allow demo responses."
            detail = self._last_error or "Model weights are missing or failed to load."
            raise RuntimeError(
                f"{self.model_key} inference requested but unavailable: {detail}. {env_hint}"
            )

    def enable_inference(self) -> None:
        """Opt in to loading the actual model weights."""

        self._fallback_mode = False

    def ensure_model_loaded(self) -> bool:
        """Lazy-load the underlying model/tokenizer when allowed.

    Returns 'True' when the model is ready for inference, otherwise
    'False' (e.g., still in fallback mode or the load failed).
        """

        if self.use_fallback():
            return False

        if self._is_initialized and self._model is not None:
            return True

        self._metrics["load_attempts"] += 1
        try:
            self._load_resources()
        except Exception as exc:  # noqa: BLE001 - propagate failure details
            self.enable_fallback(exc)
            return False

        self._is_initialized = True
        self._metrics["successful_loads"] += 1
        self._metrics["last_loaded_at"] = time.time()
        return True

    @abstractmethod
    def _load_resources(self) -> None:
        """Implementations should set 'self._model'/'self._tokenizer'."""

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------
    def build_response(self, *, content: Dict[str, Any], metadata: Optional[Dict[str, Any]]=None, warnings: Optional[Any]=None, extra: Optional[Any]=None) -> Dict[str, Any]:
        md = metadata or {}
        wrn = warnings
        if wrn is None:
            wrn_list = []
        elif isinstance(wrn, list):
            wrn_list = wrn
        else:
            wrn_list = [str(wrn)]
        extra_dict: Dict[str, Any]
        if isinstance(extra, dict):
            extra_dict = extra
        elif extra is None:
            extra_dict = {}
        else:
            # wrap scalar/list extras safely
            extra_dict = {"extra": extra}
        return {
            "model": self.model_key,
            "mode": md.get("mode","inference"),
            "content": content,
            "metadata": md,
            "warnings": wrn_list,
            **extra_dict,
        }


    def get_model_info(self) -> Dict[str, Any]:
        """Return runtime information useful for health checks."""

        return {
            "name": self.model_key,
            "mode": self.fallback_label if self.use_fallback() else self.inference_label,
            "device": str(self.device),
            "is_initialized": self._is_initialized,
            "metrics": self._metrics.copy(),
            "inference_enabled": is_inference_enabled(self.model_key),
            "last_error": self._last_error,
            "max_length": self.max_length,
    }