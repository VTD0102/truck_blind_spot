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


def test_demo_notebook_runs_app_py_instead_of_inline_pipeline() -> None:
    text = _notebook_text()

    assert '"app.py"' in text
    assert '"--device"' in text
    assert '"--no-display"' in text
    assert '["app.py", "--help"]' in text
    assert "chưa hỗ trợ --no-display" in text
    assert '"--output"' not in text
    assert "subprocess.run(" in text
    assert "BlindSpotPipeline" not in text
    assert "pipeline.process_frame" not in text


def test_demo_notebook_code_cells_are_plain_python() -> None:
    notebook = _notebook()

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"demo_notebook.ipynb cell {index}", "exec")


def test_demo_notebook_uses_colab_gpu_app_runner() -> None:
    text = _notebook_text()

    assert 'device = "0"' in text
    assert 'else "cpu"' not in text
    assert "torch.cuda.is_available()" in text
    assert "Colab chưa bật GPU" in text
    assert "xvfb-run" not in text
    assert "IPython.display" not in text
    assert "colab_app_output.mp4" not in text
    assert "stdout=subprocess.PIPE" in text
    assert "stderr=subprocess.STDOUT" in text
