"""Run Agent 5 directly from a serialized WriterInput JSON file.

Example:
    python scripts/run_agent5.py scripts/agent5_input.example.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from aca.agents.writer import WriterAgent  # noqa: E402
from aca.models.writer import WriterInput  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agent 5 without Agents 1–4.")
    parser.add_argument("input", type=Path, help="JSON file matching the WriterInput schema")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "agent5_independent_output.json",
        help="Output JSON path (Markdown is written beside it)",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        parser.error("input and output paths must be different")

    writer_input = WriterInput.model_validate_json(
        input_path.read_text(encoding="utf-8")
    )
    result = WriterAgent().write(writer_input)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(result.article_markdown.rstrip() + "\n", encoding="utf-8")
    print(f"Agent 5 JSON: {output_path}")
    print(f"Article Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
