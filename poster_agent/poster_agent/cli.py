"""Командная строка: python -m poster_agent --plan plan.md --frame frame.png --out posters/"""

from __future__ import annotations

import argparse
import sys

from .agent import PosterAgent, load_plan
from .airfail import AirFailClient
from .frame import load_frame


def main(argv=None):
    p = argparse.ArgumentParser(description="ИИ-агент: афиши по плану мероприятий в вашей рамке")
    p.add_argument("--plan", required=True, help="план мероприятий: .txt/.md/.csv/.json/.pdf/.docx/.xlsx")
    p.add_argument("--frame", required=True, help="рамка PNG с прозрачным окном (или окном цвета #00FF00)")
    p.add_argument("--out", default="posters", help="папка для результатов")
    p.add_argument("--box", help="окно вручную: x,y,w,h (пиксели)")
    p.add_argument("--no-text", action="store_true", help="картинки без надписей")
    p.add_argument("--dry-run", action="store_true", help="только промпты, без генерации картинок")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    args = p.parse_args(argv)

    box = tuple(int(v) for v in args.box.split(",")) if args.box else None
    frame = load_frame(args.frame, box=box)
    print(f"Окно в рамке: x={frame.box[0]} y={frame.box[1]} {frame.box[2]}×{frame.box[3]}")

    agent = PosterAgent(
        frame=frame,
        out_dir=args.out,
        image_client=None if args.dry_run else AirFailClient.from_env(),
        with_text=not args.no_text,
        effort=args.effort,
    )
    agent.run(load_plan(args.plan))
    ok = [r for r in agent.results if r.poster_path]
    print(f"\nГотово: {len(ok)} афиш(и) в {agent.out_dir}/ (manifest.json — все промпты и попытки)")


