"""CSV output helpers for the rebuilt result container."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .solver import SimulationResult


def write_summary_csv(path: str | Path, results: Iterable[SimulationResult]) -> None:
    rows = [result.summary for result in results]
    _write_rows(Path(path), rows)


def write_timeseries_csv(path: str | Path, results: Iterable[SimulationResult]) -> None:
    rows: list[dict[str, float | str]] = []
    for result in results:
        rows.extend(result.timeseries)
    _write_rows(Path(path), rows)


def _write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
