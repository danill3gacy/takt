"""Метрики. Импорт модулей регистрирует их в REGISTRY."""

from . import offball, players, possession, pressing, shape  # noqa: F401,E402
from .registry import REGISTRY, Computed, compute_all, metric  # noqa: F401

__all__ = ["REGISTRY", "Computed", "compute_all", "metric"]
