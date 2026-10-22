"""Export the LangGraph pipeline as Mermaid text and PNG."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aca.orchestrator.pipeline import build_pipeline

pipeline = build_pipeline()
graph = pipeline.get_graph()

# 1. Print Mermaid source
print("=" * 60)
print("MERMAID SOURCE")
print("=" * 60)
mermaid_text = graph.draw_mermaid()
print(mermaid_text)

# 2. Save Mermaid source to file
mermaid_path = ROOT / "agent_workflow.mmd"
mermaid_path.write_text(mermaid_text, encoding="utf-8")
print(f"\nMermaid source saved to: {mermaid_path}")

# 3. Try to export PNG
png_path = ROOT / "agent_workflow.png"
try:
    png_bytes = graph.draw_mermaid_png()
    png_path.write_bytes(png_bytes)
    print(f"PNG saved to: {png_path}")
except Exception as e:
    print(f"\n[PNG export skipped] {e}")
    print("To enable PNG export, install: pip install playwright && playwright install chromium")
