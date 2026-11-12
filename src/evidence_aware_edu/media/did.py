from __future__ import annotations

import time
from pathlib import Path

import requests

from evidence_aware_edu.utils import strip_evidence_markers


DID_TALKS_URL = "https://api.d-id.com/talks"


def create_did_video(
    *,
    script: str,
    api_key: str,
    presenter_url: str,
    voice_id: str = "en-GB-SoniaNeural",
    output_path: str | Path = "outputs/video.mp4",
    poll_seconds: float = 5.0,
    max_polls: int = 60,
) -> Path:
    """Render the narration as a talking-head video with D-ID."""
    headers = {
        "Authorization": f"Basic {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "source_url": presenter_url,
        "script": {
            "type": "text",
            "input": strip_evidence_markers(script),  # never speak [E-...] markers aloud
            "provider": {"type": "microsoft", "voice_id": voice_id},
        },
        "config": {"fluent": True, "pad_audio": 0.0},
    }
    response = requests.post(DID_TALKS_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    talk_id = response.json()["id"]

    result_url = None
    for _ in range(max_polls):
        status = requests.get(f"{DID_TALKS_URL}/{talk_id}", headers=headers, timeout=30)
        status.raise_for_status()
        data = status.json()
        if data.get("status") == "done" and data.get("result_url"):
            result_url = data["result_url"]
            break
        if data.get("status") in {"error", "rejected"}:
            raise RuntimeError(f"D-ID render failed: {data}")
        time.sleep(poll_seconds)

    if not result_url:
        raise TimeoutError("D-ID render did not finish before max_polls")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(result_url, stream=True, timeout=120) as video_response:
        video_response.raise_for_status()
        with output.open("wb") as f:
            for chunk in video_response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return output
