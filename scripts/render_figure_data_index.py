"""Render the human-readable panel index from its machine-readable source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "docs" / "figure_data_index.json"
MARKDOWN_PATH = ROOT / "docs" / "FIGURE_DATA_INDEX.md"


def cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def code_list(values: list[str]) -> str:
    return "<br>".join(f"`{cell(value)}`" for value in values)


def render(index: dict) -> str:
    dataset = index["dataset"]
    lines = [
        "# Figure-to-data index",
        "",
        "This panel-level index maps every panel in main Figures 1--4 and ",
        "Supplementary Figures S1--S10 to its numerical inputs, data fields, ",
        "analysis/plotting entry point, parameters, and output stem.",
        "",
        f"- Dataset version: `{dataset['version']}`",
        f"- Dataset DOI: [`{dataset['doi']}`](https://doi.org/{dataset['doi']})",
        f"- Dataset ZIP SHA-256: `{dataset['zip_sha256']}`",
        f"- Immutable source tag: `{index['source_tag']}`",
        f"- Archived path base: `{index['path_base']}`",
        "",
        "Entries beginning with `generated:` are deterministic arrays generated ",
        "by the cited source function rather than files stored in the data ZIP. ",
        "Wildcard paths identify the parameter-specific files selected by the ",
        "plotting function.",
        "",
        "| Panel | Content | Input file(s) below `results/` | Data field(s) | Script and function | Key parameters | Output stem |",
        "|---|---|---|---|---|---|---|",
    ]
    for panel in index["panels"]:
        lines.append(
            "| "
            + " | ".join((
                f"**{cell(panel['id'])}**",
                cell(panel["content"]),
                code_list(panel["inputs"]),
                code_list(panel["fields"]),
                f"`{cell(panel['code'])}`",
                cell(panel["parameters"]),
                f"`{cell(panel['output'])}`",
            ))
            + " |"
        )
    lines.extend((
        "",
        "The complete figure set is generated with:",
        "",
        "```powershell",
        "python scripts/fetch_zenodo_dataset.py",
        "python scripts/generate_paper_figures.py",
        "python scripts/verify_paper_artifacts.py",
        "```",
        "",
        "`scripts/verify_paper_artifacts.py` verifies the 14 PNG/PDF pairs and ",
        "the principal numerical values used by the manuscript.",
        "",
    ))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="fail if the Markdown file is not current")
    args = parser.parse_args()
    index = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    rendered = render(index)
    if args.check:
        current = MARKDOWN_PATH.read_text(encoding="utf-8")
        if current != rendered:
            raise SystemExit("FIGURE_DATA_INDEX.md is not current")
        print("Figure-to-data Markdown index is current")
    else:
        MARKDOWN_PATH.write_text(rendered, encoding="utf-8")
        print(f"Wrote {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
