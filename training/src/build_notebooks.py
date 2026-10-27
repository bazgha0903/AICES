"""Convert training/src/*.py (cells separated by '# %%') into training/*.ipynb"""
import json
import sys
from pathlib import Path

SRC = Path(__file__).parent
OUT = SRC.parent


def to_cells(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
        src = [l + "\n" for l in lines]
        if src:
            src[-1] = src[-1].rstrip("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


for py in sorted(list(SRC.glob("m*.py")) + list(SRC.glob("train_all*.py"))):
    nb = {"cells": to_cells(py.read_text()), "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (OUT / f"{py.stem}.ipynb").write_text(json.dumps(nb, indent=1))
    print("built", py.stem + ".ipynb", file=sys.stderr)
