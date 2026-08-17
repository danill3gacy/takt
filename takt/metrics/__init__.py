"""Метрики. Импорт модулей регистрирует их в REGISTRY."""

from .registry import REGISTRY, Computed, compute_all, metric  # noqa: F401
from . import shape, pressing, possession, offball, players  # noqa: F401,E402

__all__ = ["REGISTRY", "Computed", "compute_all", "metric"]
