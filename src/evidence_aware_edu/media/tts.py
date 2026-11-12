from __future__ import annotations

from pathlib import Path

from gtts import gTTS


def synthesize_gtts(text: str, output_path: str | Path, lang: str = "en") -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    gTTS(text=text, lang=lang).save(str(output))
    return output
