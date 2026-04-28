from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _notebook() -> dict[str, Any]:
    notebook_path = Path(__file__).resolve().parents[1] / "demo_notebook.ipynb"
    return json.loads(notebook_path.read_text(encoding="utf-8"))


def _notebook_text() -> str:
    notebook = _notebook()
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )


def test_demo_notebook_uses_current_project_defaults() -> None:
    text = _notebook_text()

    assert "weights/best_roiv2.pt" in text
    assert "assets/videos/demo4.mp4" in text
    assert "roi_profile = \"front_camera\"" in text
    assert "best_small.pt" not in text
    assert "branch = \"taitu\"" not in text


def test_demo_notebook_matches_process_frame_contract() -> None:
    text = _notebook_text()

    assert "annotated_frame, detections, tracks = pipeline.process_frame(frame)" in text
    assert "annotated_frame, _ = pipeline.process_frame(frame)" not in text


def test_demo_notebook_code_cells_are_plain_python() -> None:
    notebook = _notebook()

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"demo_notebook.ipynb cell {index}", "exec")
