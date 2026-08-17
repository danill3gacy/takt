"""Реестр метрик.

Каждая метрика объявляет, какие возможности источника ей нужны. Если фид их
не даёт — метрика не считается, а попадает в раздел отчёта «недоступно на
этом фиде». Это принципиально: лучше честно показать пробел, чем вернуть
число, посчитанное не из того.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..model import Capability, MatchSet


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    section: str
    requires: tuple[Capability, ...]
    fn: Callable[[MatchSet], Any]
    note: str = ""


REGISTRY: dict[str, MetricSpec] = {}


def metric(key: str, title: str, section: str,
           requires: tuple[Capability, ...] = (), note: str = ""):
    def deco(fn):
        REGISTRY[key] = MetricSpec(key, title, section, requires, fn, note)
        return fn
    return deco


@dataclass
class Computed:
    values: dict[str, Any] = field(default_factory=dict)
    unavailable: list[tuple[str, list[str]]] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def get(self, key: str, default=None) -> Any:
        return self.values.get(key, default)


def compute_all(ms: MatchSet, strict: bool = False) -> Computed:
    out = Computed()
    caps = ms.capabilities
    for key, spec in REGISTRY.items():
        missing = [c.value for c in spec.requires if c not in caps]
        if missing:
            out.unavailable.append((spec.title, missing))
            continue
        try:
            out.values[key] = spec.fn(ms)
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise
            out.failed.append((spec.title, f"{type(exc).__name__}: {exc}"))
    return out
