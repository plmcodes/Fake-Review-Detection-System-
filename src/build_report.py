"""Copy the generated results tables into report/report.md and README.md.

    python -m src.build_report

The tables in results/tables/*.md are produced by `python -m src.evaluate` from results/runs.csv.
This script only pastes them between markers, so no number in the report or README is typed by hand:

    report/report.md   <!-- TABLE:<name> --> ... <!-- /TABLE -->
    README.md          <!-- RESULTS:START --> ... <!-- RESULTS:END -->
"""
from __future__ import annotations

import re

from .config import resolve_path
from .utils import get_logger

log = get_logger("build_report")

README_TABLES = ["main_comparison", "cross_domain", "ablation", "modern_fakes"]


def table_body(name: str) -> str:
    """The table file without its '## title' line (the report has its own headings)."""
    path = resolve_path(f"results/tables/{name}.md")
    if not path.exists():
        return "TBD - not run yet."
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if lines and lines[0].startswith("## "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def fill_report() -> None:
    path = resolve_path("report/report.md")
    text = path.read_text(encoding="utf-8")

    def replace(match: re.Match) -> str:
        name = match.group(1)
        return f"<!-- TABLE:{name} -->\n{table_body(name)}\n<!-- /TABLE -->"

    new = re.sub(r"<!-- TABLE:(\w+) -->.*?<!-- /TABLE -->", replace, text, flags=re.S)
    path.write_text(new, encoding="utf-8")
    log.info("filled tables in %s", path)


def fill_readme() -> None:
    path = resolve_path("README.md")
    text = path.read_text(encoding="utf-8")
    parts = ["<!-- RESULTS:START -->", "## Results", "",
             "Generated from `results/runs.csv` (`python -m src.evaluate --tables`, then `python -m src.build_report`). "
             "Discussion: [`report/report.md`](report/report.md).", ""]
    titles = {"main_comparison": "Main comparison (Salminen test split)",
              "cross_domain": "Cross-domain: train on Amazon (Salminen), test on hotels (Ott)",
              "ablation": "Ablation of the hybrid model (Model D)",
              "modern_fakes": "Fakes written by a modern language model"}
    for name in README_TABLES:
        parts += [f"### {titles[name]}", "", table_body(name), ""]
    parts.append("<!-- RESULTS:END -->")
    new = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", "\n".join(parts), text, flags=re.S)
    path.write_text(new, encoding="utf-8")
    log.info("filled results in %s", path)


def main():
    fill_report()
    fill_readme()


if __name__ == "__main__":
    main()
