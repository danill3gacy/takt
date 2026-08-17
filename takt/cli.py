"""Командная строка.

    python -m takt.cli report --team "Auckland FC" --out out/report.html

В боевом контуре сюда добавляется `--source yandex --match-ids ...`, и всё
остальное не меняется.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from .baseline import build_baseline
from .model import MatchSet
from .render import render_html
from .report import build_report
from .sources.skillcorner import SkillCornerSource

SOURCES = {"skillcorner": SkillCornerSource}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="takt")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("report", help="собрать отчёт по сопернику")
    p.add_argument("--source", default="skillcorner", choices=list(SOURCES))
    p.add_argument("--data", default="data/skillcorner/data")
    p.add_argument("--team", required=True, help="разбираемая команда (соперник)")
    p.add_argument("--club", default="ФК «Динамо» Москва")
    p.add_argument("--club-short", default="Д")
    p.add_argument("--out", default="out/report.html")
    p.add_argument("--date", default=None, help="дата отчёта, по умолчанию сегодня")

    q = sub.add_parser("matches", help="показать доступные матчи")
    q.add_argument("--source", default="skillcorner", choices=list(SOURCES))
    q.add_argument("--data", default="data/skillcorner/data")
    q.add_argument("--team", default=None)

    args = ap.parse_args(argv)
    src = SOURCES[args.source](args.data)

    if args.cmd == "matches":
        for r in src.list_matches(args.team):
            print(f"{r['id']}  {r['date']}  {r['home']} — {r['away']}")
        return 0

    tid = src.team_id_by_name(args.team)
    all_matches = src.load_many([r["id"] for r in src.list_matches()])
    subject = [m for m in all_matches if tid in (m.home.id, m.away.id)]
    if not subject:
        raise SystemExit(f"нет матчей команды {args.team}")
    team = next(m.home if m.home.id == tid else m.away for m in subject)

    print(f"матчей у команды: {len(subject)}; считаю лига-бенчмарк...")
    baseline = build_baseline(all_matches)
    print(f"бенчмарк: {baseline.n_teams} команд, {baseline.n_matches} матчей")

    date = args.date or dt.date.today().strftime("%d.%m.%Y")
    rep = build_report(MatchSet(team, subject), baseline,
                       club=args.club, club_short=args.club_short, generated=date)
    out = render_html(rep, args.out)
    print(f"тезисов: {len(rep.insights)}, паттернов: {len(rep.patterns)}")
    print(f"готово: {Path(out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
