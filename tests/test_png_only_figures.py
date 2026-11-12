from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


def test_figure_runner_writes_pngs_and_clears_old_pdf_figures(monkeypatch, tmp_path):
    # Older experiments may already have PDF figure copies from previous versions.
    experiment_dir = tmp_path / "results" / "experiments" / "EXP_PNG_TEST"
    figures_dir = experiment_dir / "figures"
    figures_dir.mkdir(parents=True)
    (figures_dir / "Figure_01_study_design.pdf").write_bytes(b"old figure")

    project_root = Path(__file__).resolve().parents[1]
    scripts_dir = project_root / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    spec = importlib.util.spec_from_file_location("png_only_figures", scripts_dir / "generate_research_figures.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(sys, "argv", ["generate_research_figures.py", "--experiment-id", "EXP_PNG_TEST"])
    module.main()

    pngs = sorted(figures_dir.glob("Figure_*.png"))
    assert len(pngs) == 3
    assert len(list(figures_dir.glob("Architecture_*.png"))) == 6
    assert not (figures_dir / "Figure_02_proposed_architecture.png").exists()
    assert not list(figures_dir.glob("*.pdf"))
