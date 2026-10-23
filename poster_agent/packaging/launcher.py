"""Точка входа exe: двойной клик открывает окно, с аргументами работает как CLI."""
import os
import sys

# У оконного exe на Windows нет консоли: stdout/stderr = None, и print упал бы.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

from poster_agent.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
