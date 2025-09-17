#!/usr/bin/env python3
"""
Lightweight Weights & Biases (wandb) wrapper that fails gracefully when wandb
isn't installed or explicitly disabled. Use to log metrics, configs, and
optionally watch the model gradients/parameters.
"""

from __future__ import annotations

import os
import math
from typing import Any, Dict, Optional


class WandBLogger:
    def __init__(
        self,
        enabled: Optional[bool] = None,
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        entity: Optional[str] = None,
        mode: Optional[str] = None,  # e.g., 'offline'
    ) -> None:
        # Determine enabled state: default to enabled unless WANDB_DISABLED=true
        if enabled is None:
            disabled_env = os.getenv("WANDB_DISABLED", "false").lower() in ("1", "true", "yes")
            enabled = not disabled_env

        self.enabled = bool(enabled)
        self._wandb = None
        self._active = False

        if not self.enabled:
            return

        try:
            import wandb  # type: ignore

            kwargs = {
                "project": project or os.getenv("WANDB_PROJECT", "yxanul"),
                "name": run_name or os.getenv("WANDB_RUN_NAME"),
                "entity": entity or os.getenv("WANDB_ENTITY"),
                "config": config or {},
            }
            # Support explicit WANDB_MODE=offline/online
            _mode = mode or os.getenv("WANDB_MODE")
            if _mode is not None:
                kwargs["mode"] = _mode

            self._wandb = wandb
            self._wandb.init(**kwargs)
            self._active = True
        except Exception as e:  # pragma: no cover
            # Fall back to no-op if anything goes wrong
            print(f"[wandb] disabled: {e}")
            self.enabled = False
            self._wandb = None
            self._active = False

    def watch(self, model, log: str = "gradients", log_freq: int = 200) -> None:
        if self._active:
            try:
                self._wandb.watch(model, log=log, log_freq=log_freq)
            except Exception:
                pass

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._active:
            try:
                self._wandb.log(metrics, step=step)
            except Exception:
                pass

    def log_eval(self, losses: Dict[str, Any], step: int) -> None:
        """Log eval metrics (loss and perplexity per split)."""
        if not self._active:
            return
        payload: Dict[str, Any] = {}
        for split, loss_val in losses.items():
            try:
                lv = float(loss_val)
            except Exception:
                continue
            payload[f"{split}/loss"] = lv
            try:
                payload[f"{split}/ppl"] = math.exp(min(20.0, lv))
            except Exception:
                pass
        if payload:
            self.log_metrics(payload, step=step)

    def set_summary(self, **kwargs: Any) -> None:
        if self._active:
            try:
                for k, v in kwargs.items():
                    self._wandb.run.summary[k] = v
            except Exception:
                pass

    def finish(self) -> None:
        if self._active:
            try:
                self._wandb.finish()
            except Exception:
                pass

