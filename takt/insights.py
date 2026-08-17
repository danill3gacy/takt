"""Генератор тезисов.

Здесь проведена граница, на которой держится доверие штаба к системе:

  ФАКТ      — арифметика над данными. Система не может ошибиться, потому что
              ошибаться нечему: столько-то эпизодов, столько-то метров.
  ГИПОТЕЗА  — тактическая интерпретация факта. Формулируется как «в N эпизодах
              из M это выглядит так — вот все N», а не как утверждение.

Ни один тезис не выпускается без списка эпизодов, которые его подтверждают.
Тренер должен иметь возможность проверить любую строку отчёта за два клика —
это единственное, что отличает инструмент от гадания.

Вес тезиса = насколько сильно отклонение от лиги × сколько под ним наблюдений.
Тезисы без веса не показываются: длинный отчёт, где всё «важно», не читают.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .baseline import Baseline
from .metrics import Computed


@dataclass
class Insight:
    id: str
    text: str
    kind: str  # "факт" | "гипотеза"
    section: str
    weight: float = 0.0  # 0..100
    evidence: str = ""  # на чём основано
    n: int = 0  # число наблюдений
    comparison: dict | None = None  # сравнение с лигой
    clips: list[dict] = field(default_factory=list)
    detail: str = ""

    @property
    def level(self) -> str:
        if self.weight >= 66:
            return "high"
        if self.weight >= 38:
            return "mid"
        return "low"


def _weight(z: float | None, n: int, cap: int = 120) -> float:
    """Вес = сила отклонения × уверенность от объёма выборки."""
    strength = min(abs(z or 0.0) / 2.0, 1.0)
    confidence = min(math.log1p(n) / math.log1p(cap), 1.0)
    return round(100 * (0.65 * strength + 0.35 * confidence), 1)


def _get(c: Computed, key: str, *path: str, default: Any = None) -> Any:
    v = c.get(key)
    for p in path:
        if not isinstance(v, dict):
            return default
        v = v.get(p)
    return v if v is not None else default


def build_insights(c: Computed, bl: Baseline, team_name: str) -> list[Insight]:
    out: list[Insight] = []

    # ---------------------------------------------------------------- оборона
    forms = c.get("formations") or {}
    if forms.get("rows"):
        top = forms["rows"][0]
        n = int(forms.get("n") or 0)
        out.append(
            Insight(
                id="formation",
                text=f"Обороняется в {top['key']} — {top['share']}% зафиксированных эпизодов",
                kind="факт",
                section="Оборона",
                weight=_weight((top["share"] - 40) / 15, n),
                evidence=f"расстановка распознана по позициям игроков в {n} эпизодах владения соперника",
                n=n,
                clips=forms.get("clips", []),
                detail="Прочие схемы: "
                + ", ".join(f"{r['key']} ({r['share']}%)" for r in forms["rows"][1:4]),
            )
        )

    dl = c.get("defensive_line") or {}
    if dl.get("mean") is not None:
        cmp_ = bl.compare("line_height", dl["mean"])
        n = sum(r["n"] for r in dl.get("by_phase", []))
        out.append(
            Insight(
                id="line_height",
                text=f"Средняя высота последней линии — {dl['mean']} м ({cmp_['rank'] if cmp_ else '—'})",
                kind="факт",
                section="Оборона",
                weight=_weight(cmp_["z"] if cmp_ else 0, n),
                evidence=f"{n} измерений в моменты владения соперника; медиана лиги "
                f"{cmp_['median'] if cmp_ else '—'} м",
                n=n,
                comparison=cmp_,
                clips=dl.get("clips", []),
                detail="; ".join(
                    f"{r['phase']} — {r['height']} м" for r in dl.get("by_phase", [])[:4]
                ),
            )
        )
        bt = dl.get("by_time") or []
        if len(bt) >= 4:
            first = bt[0]["height"]
            last = bt[-1]["height"]
            drop = first - last
            if abs(drop) >= 2:
                direction = "опускается" if drop > 0 else "поднимается"
                out.append(
                    Insight(
                        id="line_drift",
                        text=f"К концу матча линия {direction} на {abs(round(drop, 1))} м "
                        f"({first} → {last})",
                        kind="гипотеза",
                        section="Оборона",
                        weight=_weight(drop / 3.0, sum(r["n"] for r in bt)),
                        evidence=f"среднее по 15-минутным отрезкам, {sum(r['n'] for r in bt)} измерений",
                        n=sum(r["n"] for r in bt),
                        clips=dl.get("clips_late", []),
                        detail="Возможные причины разные — усталость, счёт, замены. "
                        "Проверять на эпизодах, а не принимать как данность.",
                    )
                )

    bs = c.get("block_shape") or {}
    if bs.get("rows"):
        cmp_w = bl.compare("block_width", bs.get("mean_width"))
        low = next((r for r in bs["rows"] if r["phase"] == "низкий блок"), None)
        n = int(bs.get("n_phases_out") or 0)
        txt = f"Блок без мяча — {bs['mean_width']} м в ширину и {bs['mean_length']} м в длину"
        if low:
            txt += f"; в низком блоке сужается до {low['width']} м"
        out.append(
            Insight(
                id="block",
                text=txt,
                kind="факт",
                section="Оборона",
                weight=_weight(cmp_w["z"] if cmp_w else 0, n),
                evidence=f"{n} оборонительных фаз; медиана лиги по ширине "
                f"{cmp_w['median'] if cmp_w else '—'} м",
                n=n,
                comparison=cmp_w,
                clips=bs.get("clips", []),
                detail="; ".join(
                    f"{r['phase']}: {r['width']}×{r['length']} м" for r in bs["rows"][:4]
                ),
            )
        )
        if low and low["width"] < 33:
            out.append(
                Insight(
                    id="narrow_block",
                    text=f"В низком блоке ширина падает до {low['width']} м — фланги отдаются",
                    kind="гипотеза",
                    section="Оборона",
                    weight=_weight((33 - low["width"]) / 2.0, low["n"]),
                    evidence=f"{low['n']} фаз низкого блока со средней шириной {low['width']} м "
                    f"при ширине поля 68 м",
                    n=low["n"],
                    clips=bs.get("clips_low", []),
                    detail="Разница между шириной поля и шириной блока — это и есть свободная зона "
                    "для смены фланга. Смотреть, успевает ли дальний защитник смещаться.",
                )
            )

    pp = c.get("pressing_profile") or {}
    ppda_v = _get(c, "ppda", "value")
    if ppda_v is not None:
        cmp_ = bl.compare("ppda", ppda_v)
        n = int(pp.get("total") or 0)
        out.append(
            Insight(
                id="ppda",
                text=f"PPDA {ppda_v} — {cmp_['rank'] if cmp_ else 'нет базы для сравнения'}",
                kind="факт",
                section="Оборона",
                weight=_weight(-(cmp_["z"]) if cmp_ else 0, n),
                evidence=f"{n} оборонительных действий выше своей трети; медиана лиги "
                f"{cmp_['median'] if cmp_ else '—'}",
                n=n,
                comparison=cmp_,
                clips=(c.get("ppda") or {}).get("clips", []),
                detail="Меньше значение — чаще вступают в отбор, не давая сопернику развивать атаку.",
            )
        )
    if pp.get("thirds"):
        top_third = max(pp["thirds"], key=lambda r: r["n"])
        out.append(
            Insight(
                id="press_zone",
                text=f"Вступают в отбор в основном в зоне «{top_third['key']}» — "
                f"{top_third['share']}% действий",
                kind="факт",
                section="Оборона",
                weight=_weight((top_third["share"] - 40) / 12, top_third["n"]),
                evidence=f"{pp['total']} оборонительных действий за {bl.n_matches and ''}выборку",
                n=top_third["n"],
                clips=pp.get("clips", []),
                detail="По каналам: "
                + ", ".join(f"{r['key']} {r['share']}%" for r in pp.get("channels", [])),
            )
        )

    rz = c.get("regain_zones") or {}
    if rz.get("high_share") is not None:
        cmp_ = bl.compare("high_regains", rz["high_share"])
        out.append(
            Insight(
                id="high_regain",
                text=f"{rz['high_share']}% возвратов мяча — на чужой половине "
                f"({rz['high_regains']} из {rz['total']})",
                kind="факт",
                section="Оборона",
                weight=_weight(cmp_["z"] if cmp_ else 0, rz["total"]),
                evidence=f"{rz['total']} возвратов владения в выборке",
                n=int(rz["total"]),
                comparison=cmp_,
                clips=rz.get("clips", []),
            )
        )

    pc = c.get("pressing_chains") or {}
    if pc.get("regain_share") is not None:
        out.append(
            Insight(
                id="press_chains",
                text=f"Цепочки прессинга заканчиваются отбором в {pc['regain_share']}% случаев, "
                f"средняя длина цепочки — {pc.get('mean_length')} действий",
                kind="факт",
                section="Оборона",
                weight=_weight((pc["regain_share"] - 30) / 10, int(pc.get("n_chains") or 0)),
                evidence=(
                    f"{pc.get('n_chains')} цепочек, "
                    f"максимальная — {pc.get('max_length')} действий подряд"
                ),
                n=int(pc.get("n_chains") or 0),
                clips=pc.get("clips", []),
            )
        )

    # ----------------------------------------------------------------- атака
    pr = c.get("possession_profile") or {}
    if pr.get("possession_share") is not None:
        cmp_ = bl.compare("possession", pr["possession_share"])
        out.append(
            Insight(
                id="possession",
                text=f"Владение {pr['possession_share']}%, средняя фаза атаки — "
                f"{pr['mean_phase_duration']} с ({cmp_['rank'] if cmp_ else '—'})",
                kind="факт",
                section="Атака",
                weight=_weight(cmp_["z"] if cmp_ else 0, int(pr.get("own_touches") or 0)),
                evidence=f"{pr.get('n_phases')} фаз атаки, {pr.get('own_touches')} владений мячом",
                n=int(pr.get("own_touches") or 0),
                comparison=cmp_,
                clips=pr.get("clips", []),
                detail="; ".join(
                    f"{r['phase']} {r['share']}% (удар в {r['shot_rate']}%)"
                    for r in pr.get("rows", [])[:4]
                ),
            )
        )

    pg = c.get("progression") or {}
    if pg.get("rows"):
        cmp_ = bl.compare("breaks", pg.get("per_match"))
        thr = [r for r in pg["rows"] if r["line"] == "предпоследняя"]
        out.append(
            Insight(
                id="progression",
                text=f"{pg['per_match']} взломов линий за матч; чаще всего вскрывают "
                f"предпоследнюю линию"
                if thr
                else f"{pg['per_match']} взломов линий за матч",
                kind="факт",
                section="Атака",
                weight=_weight(cmp_["z"] if cmp_ else 0, int(pg.get("total") or 0)),
                evidence=f"{pg.get('total')} взломов; в среднем обыгрывают "
                f"{pg.get('mean_bypassed')} соперника за действие",
                n=int(pg.get("total") or 0),
                comparison=cmp_,
                clips=pg.get("clips", []),
                detail="; ".join(
                    f"{r['line']}: {r['n']} (через {r['through']}, вокруг {r['around']})"
                    for r in pg["rows"]
                ),
            )
        )
        ch = pg.get("channels") or []
        if ch:
            top = max(ch, key=lambda r: r["n"])
            others = sum(r["n"] for r in ch) - top["n"]
            if top["share"] >= 33 and top["n"] >= 8:
                out.append(
                    Insight(
                        id="progression_side",
                        text=f"Последнюю линию вскрывают в основном через «{top['key']}» — "
                        f"{top['share']}% случаев",
                        kind="гипотеза",
                        section="Атака",
                        weight=_weight((top["share"] - 20) / 10, top["n"]),
                        evidence=f"{top['n']} взломов последней линии из {top['n'] + others}",
                        n=top["n"],
                        clips=pg.get("clips", []),
                        detail="Асимметрия может быть и следствием соперников выборки — "
                        "проверить, повторяется ли она от матча к матчу.",
                    )
                )

    tm = c.get("threat_map") or {}
    if tm.get("total") is not None:
        cmp_ = bl.compare("threat", tm["total"])
        out.append(
            Insight(
                id="threat",
                text=f"Создают {tm['total']} xT за матч, пропускают {tm.get('conceded')} "
                f"({cmp_['rank'] if cmp_ else '—'})",
                kind="факт",
                section="Атака",
                weight=_weight(cmp_["z"] if cmp_ else 0, len(tm.get("clips", [])) * 4 or 40),
                evidence="сумма модельной угрозы по всем передачам, нормировано на матч",
                n=int((c.get("possession_profile") or {}).get("own_touches") or 0),
                comparison=cmp_,
                clips=tm.get("clips", []),
                detail="По каналам: "
                + ", ".join(f"{r['key']} {r['value']}" for r in tm.get("by_channel", [])),
            )
        )
    if tm.get("potential_taken") is not None:
        cmp_ = bl.compare("threat_taken", tm["potential_taken"])
        out.append(
            Insight(
                id="threat_taken",
                text=f"Из доступной угрозы используют {tm['potential_taken']}% — "
                f"игрок с мячом чаще выбирает не самый опасный вариант",
                kind="гипотеза",
                section="Атака",
                weight=_weight(
                    -(cmp_["z"]) if cmp_ else 0,
                    int((c.get("passing_options") or {}).get("n_options") or 0),
                ),
                evidence=f"суммарный xT всех открытых вариантов — {tm.get('potential')} за матч, "
                f"реализовано {tm.get('total')}",
                n=int((c.get("passing_options") or {}).get("n_options") or 0),
                comparison=cmp_,
                clips=(c.get("passing_options") or {}).get("clips", []),
                detail="Разрыв показывает, где команда недобирает: партнёры открываются, "
                "но передача уходит в безопасную сторону.",
            )
        )

    ls = c.get("losses") or {}
    if ls.get("own_third_share") is not None and ls["own_third_share"] >= 22:
        out.append(
            Insight(
                id="losses_own_third",
                text=f"{ls['own_third_share']}% потерь — в своей трети "
                f"({ls['own_third']} из {ls['total']})",
                kind="факт",
                section="Атака",
                weight=_weight((ls["own_third_share"] - 20) / 6, int(ls["total"])),
                evidence=f"{ls['total']} потерь в выборке, {ls['per_match']} за матч",
                n=int(ls["total"]),
                clips=ls.get("clips", []),
                detail="Чаще прочих теряют: "
                + ", ".join(f"{r['name']} ({r['n']})" for r in ls.get("top_players", [])[:3]),
            )
        )

    # ------------------------------------------------------ движение без мяча
    ob = c.get("off_ball_runs") or {}
    if ob.get("per_match"):
        cmp_ = bl.compare("runs", ob["per_match"])
        cmp_u = bl.compare("runs_used", ob.get("used_share"))
        out.append(
            Insight(
                id="runs",
                text=f"{ob['per_match']} забеганий за матч, использовано {ob['used_share']}% "
                f"({cmp_['rank'] if cmp_ else '—'})",
                kind="факт",
                section="Движение без мяча",
                weight=_weight(cmp_["z"] if cmp_ else 0, int(ob.get("total") or 0)),
                evidence=f"{ob.get('total')} забеганий в выборке; медиана лиги "
                f"{cmp_['median'] if cmp_ else '—'} за матч",
                n=int(ob.get("total") or 0),
                comparison=cmp_ or cmp_u,
                clips=ob.get("clips_all", []),
                detail="; ".join(
                    f"{r['kind']} {r['n']} (исп. {r['used']}%)" for r in ob.get("rows", [])[:4]
                ),
            )
        )
    if ob.get("behind_per_match") is not None:
        cmp_ = bl.compare("behind", ob["behind_per_match"])
        used = ob.get("behind_used")
        out.append(
            Insight(
                id="behind",
                text=f"Забегания за спину — {ob['behind_per_match']} за матч, "
                f"из них адресовано {used}%",
                kind="факт" if used is None else "гипотеза",
                section="Движение без мяча",
                weight=_weight(cmp_["z"] if cmp_ else 0, int(ob.get("behind") or 0)),
                evidence=f"{ob.get('behind')} забеганий за линию обороны в выборке",
                n=int(ob.get("behind") or 0),
                comparison=cmp_,
                clips=ob.get("clips", []),
                detail="Ломают линию обороны движением в "
                + str(ob.get("break_line"))
                + " эпизодах, отодвигают её назад в "
                + str(ob.get("push_line"))
                + ".",
            )
        )

    po = c.get("passing_options") or {}
    if po.get("mean_options") is not None:
        cmp_ = bl.compare("options", po["mean_options"])
        out.append(
            Insight(
                id="options",
                text=f"У игрока с мячом в среднем {po['mean_options']} открытых партнёров, "
                f"из них {po.get('mean_forward_options')} впереди",
                kind="факт",
                section="Движение без мяча",
                weight=_weight(cmp_["z"] if cmp_ else 0, int(po.get("n_options") or 0)),
                evidence=f"{po.get('n_options')} зафиксированных вариантов передачи",
                n=int(po.get("n_options") or 0),
                comparison=cmp_,
                clips=po.get("clips", []),
                detail=f"Опасный вариант выбирают в {po.get('dangerous_taken')}% случаев, "
                f"безопасный — в {po.get('safe_taken')}%.",
            )
        )

    # ------------------------------------------------------------- стандарты
    sp = c.get("set_pieces") or {}
    for row in sp.get("rows", []):
        if row["kind"] == "угловой" and row["n"] >= 5:
            out.append(
                Insight(
                    id="corners",
                    text=f"Угловые: {row['per_match']} за матч, удар следует в {row['shot_rate']}%",
                    kind="факт",
                    section="Стандарты",
                    weight=_weight((row["shot_rate"] - 30) / 15, row["n"]),
                    evidence=f"{row['n']} угловых в выборке",
                    n=row["n"],
                    clips=sp.get("clips", []),
                )
            )

    # ------------------------------------------------------------------ ритм
    tp = c.get("time_profile") or {}
    rows = tp.get("rows") or []
    if len(rows) == 6:
        best = max(rows, key=lambda r: r["xthreat"])
        worst = max(rows, key=lambda r: r["xthreat_against"])
        tot = sum(r["xthreat"] for r in rows) or 1
        if best["xthreat"] / tot >= 0.22:
            out.append(
                Insight(
                    id="peak",
                    text=f"Пик угрозы — отрезок {best['bucket']} мин: {best['xthreat']} xT "
                    f"({round(100 * best['xthreat'] / tot)}% всей угрозы за матч)",
                    kind="факт",
                    section="Ритм",
                    weight=_weight((100 * best["xthreat"] / tot - 16.7) / 6, int(best["shots"])),
                    evidence=f"{best['shots']} ударов в этом отрезке; распределение считалось "
                    f"по всем матчам выборки",
                    n=int(best["shots"]),
                    clips=tp.get("clips_peak", []),
                    detail="; ".join(
                        f"{r['bucket']}: {r['xthreat']} xT / {r['shots']} уд." for r in rows
                    ),
                )
            )
        tot_a = sum(r["xthreat_against"] for r in rows) or 1
        if worst["xthreat_against"] / tot_a >= 0.24:
            out.append(
                Insight(
                    id="vuln",
                    text=f"Больше всего пропускают угрозы в отрезке {worst['bucket']} мин: "
                    f"{worst['xthreat_against']} xT против",
                    kind="гипотеза",
                    section="Ритм",
                    weight=_weight(
                        (100 * worst["xthreat_against"] / tot_a - 16.7) / 6,
                        int(worst["shots_against"]),
                    ),
                    evidence=f"{worst['shots_against']} ударов соперников в этом отрезке",
                    n=int(worst["shots_against"]),
                    clips=tp.get("clips_vuln", []),
                    detail="Отрезок стоит проверить отдельно: провал может быть связан "
                    "с заменами, счётом или конкретным соперником.",
                )
            )

    # ----------------------------------------------------------------- игроки
    kp = c.get("key_players") or {}
    for prof in kp.get("profiles", []):
        out.append(
            Insight(
                id=f"player_{prof['id']}",
                text=f"{prof['name']} — {prof['role'].lower()} ({prof['per90']} за 90 мин)",
                kind="факт",
                section="Игроки",
                weight=_weight(1.0, int(prof.get("minutes") or 0)),
                evidence=f"{prof.get('minutes')} минут в выборке",
                n=int(prof.get("minutes") or 0),
                clips=prof.get("clips", []),
            )
        )

    # Принцип продукта: утверждение без эпизодов не показывается. Если под
    # тезисом нечего смотреть, проверить его нельзя — значит, его нет.
    out = [i for i in out if i.clips]
    out.sort(key=lambda i: -i.weight)
    return out
