"""Запуск: без аргументов — окно приложения, с аргументами — командная строка.

    python -m poster_agent
    python -m poster_agent --plan plan.md --frame frame.png --out posters/
"""

from __future__ import annotations

import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        from .gui import main as gui_main

        return gui_main()
    from .cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
