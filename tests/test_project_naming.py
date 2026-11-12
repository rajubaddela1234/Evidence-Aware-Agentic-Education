from pathlib import Path


def test_retired_name_not_present_in_project_text():
    retired = "trace" + "-vid"
    retired_alt = "trace" + "_vid"
    retired_pkg = "trace" + "vid"
    for path in Path(".").rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            continue
        if path.suffix.lower() not in {".py", ".md", ".toml", ".jsonl", ".example", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        assert retired not in text, str(path)
        assert retired_alt not in text, str(path)
        assert retired_pkg not in text, str(path)
