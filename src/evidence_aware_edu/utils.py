from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, TypeVar

T = TypeVar("T")


def stable_id(prefix: str, *parts: str) -> str:
    raw = "||".join(parts).encode("utf-8", errors="ignore")
    return f"{prefix}-{hashlib.sha1(raw).hexdigest()[:12]}"


_ONE_MARKER = r"\[E-[\w-]+(?:\s*[,;]\s*E-[\w-]+)*\]"
# A run of markers such as "[E-a], [E-b]; [E-c]" is removed as one unit, separators included,
# so stripping does not leave stray ", ," fragments in the spoken text.
_EVIDENCE_MARKER = re.compile(
    rf"[ \t]*\(?{_ONE_MARKER}(?:[ \t]*(?:[,;]|and)?[ \t]*{_ONE_MARKER})*\)?"
)
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([.,;:!?])")
_REPEATED_COMMA = re.compile(r",(?:[ \t]*,)+")


def strip_evidence_markers(text: str) -> str:
    """Remove ``[E-...]`` verification markers so the text reads as spoken narration.

    Markers stay in the stored script for claim/citation verification; narration judges
    and the video renderer should receive this cleaned text instead.
    """
    text = _EVIDENCE_MARKER.sub("", text or "")
    text = _REPEATED_COMMA.sub(",", text)
    return _SPACE_BEFORE_PUNCT.sub(r"\1", text)


_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# Footnote links such as "[[1]](#cite_note-IBM-1)" and bare "[12]" references.
_FOOTNOTE = re.compile(r"\[\[[^\]]{0,20}\]\]\([^)]*\)|\[\d{1,3}\]")
_MD_LINK = re.compile(r"\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)")
_HTML_TAG = re.compile(r"<[^>]{1,200}>")
_BARE_URL = re.compile(r"https?://\S+")
_LIST_OR_HEADING_PREFIX = re.compile(r"^\s*(?:[#>*+\-|]+|\d+[.)])\s*")


def clean_web_text(text: str) -> str:
    """Remove navigation, images and link markup from scraped web markdown.

    Pages fetched as raw markdown start with menus, logos and link lists ("[Jump to content]",
    "![Wikipedia](...)"). Those lines crowd out real content in passage ranking. Lines that are
    mostly link markup or too short to be prose are dropped; link anchor text inside real
    sentences is kept.
    """
    out: list[str] = []
    for line in (text or "").splitlines():
        line = _MD_IMAGE.sub("", line)
        line = _FOOTNOTE.sub("", line)
        line = _HTML_TAG.sub(" ", line)
        if _MD_LINK.search(line):
            # Text left after deleting links entirely: menus and link lists have almost none.
            remainder = _LIST_OR_HEADING_PREFIX.sub("", _MD_LINK.sub("", line)).strip(" |·•-")
            if len(remainder) < 25:
                continue
            line = _MD_LINK.sub(r"\1", line)
        line = _BARE_URL.sub("", line).replace("\xa0", " ").rstrip()
        content = _LIST_OR_HEADING_PREFIX.sub("", line).strip()
        is_heading = line.lstrip().startswith("#")
        if not content:
            out.append("")
            continue
        if not is_heading and len(content.split()) < 4:
            continue
        out.append(line)
    cleaned = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def clip(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def dedupe_by(items: Iterable[T], key_fn) -> list[T]:
    seen: set[str] = set()
    out: list[T] = []
    for item in items:
        key = str(key_fn(item)).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def json_dump(path: Path, data) -> None:
    """Write JSON through a temporary file so partial files are not left behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        f.flush()
    tmp.replace(path)


def json_load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_iso_date(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
