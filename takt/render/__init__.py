"""Рендер отчёта в самодостаточный HTML."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..report import Report

TEMPLATES = Path(__file__).parent / "templates"


def _clean(obj: Any) -> Any:
    """NaN/Inf в JSON недопустимы — превращаем в null, иначе JSON.parse упадёт."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if hasattr(obj, "item"):  # numpy-скаляры
        try:
            return _clean(obj.item())
        except Exception:  # noqa: BLE001
            return str(obj)
    return obj


def render_html(report: Report, out_path: str | Path) -> Path:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        autoescape=select_autoescape(["html"]),
    )
    tpl = env.get_template("report.html.j2")
    data = _clean(report.to_dict())
    html = tpl.render(
        r=report,
        data_json=json.dumps(data, ensure_ascii=False, allow_nan=False).replace(
            "</script>", "<\\/script>"
        ),
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
