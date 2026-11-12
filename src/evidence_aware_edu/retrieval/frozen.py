from __future__ import annotations

from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from evidence_aware_edu.schemas import SourceDocument, SourceType
from evidence_aware_edu.utils import clean_web_text, json_load


class FrozenCorpusHub:
    """Search a saved reference pool using local TF-IDF ranking."""

    def __init__(self, reference_pool_dir: Path, require_video_transcript: bool = True) -> None:
        self.reference_pool_dir = reference_pool_dir
        self.require_video_transcript = require_video_transcript

    def load(self, prompt_id: str) -> list[SourceDocument]:
        path = self.reference_pool_dir / f"{prompt_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing frozen reference pool: {path}. Run scripts/build_reference_pool.py first."
            )
        raw = json_load(path)
        docs = [SourceDocument.model_validate(x) for x in raw["documents"]]
        # Saved web pages are raw markdown; strip menus/images/link lists at load time so the
        # stored pool stays byte-identical (its hash is part of the experiment identity).
        for doc in docs:
            if doc.source_type == "web":
                cleaned = clean_web_text(doc.text)
                if cleaned:
                    doc.text = cleaned
        if self.require_video_transcript:
            # Ignore old video entries that were saved without transcripts.
            docs = [
                d
                for d in docs
                if not (
                    d.source_type == "video"
                    and d.metadata.get("has_transcript") is not True
                )
            ]
        return docs

    @staticmethod
    def _index_text(doc: SourceDocument) -> str:
        # Search the whole paper; cap web/video text to keep ranking light.
        body = doc.text if doc.source_type == "academic" else doc.text[:12000]
        return f"{doc.title}\n{body}"

    @staticmethod
    def rank(
        query: str,
        documents: Iterable[SourceDocument],
        source_types: list[SourceType],
        max_per_source: int,
        per_source_limits: dict[SourceType, int] | None = None,
    ) -> list[SourceDocument]:
        candidates = [d for d in documents if d.source_type in source_types]
        if not candidates:
            return []
        texts = [query] + [FrozenCorpusHub._index_text(d) for d in candidates]
        try:
            matrix = TfidfVectorizer(
                stop_words="english", ngram_range=(1, 2), min_df=1
            ).fit_transform(texts)
            scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        except ValueError:
            scores = [0.0] * len(candidates)

        grouped: dict[str, list[tuple[float, SourceDocument]]] = {s: [] for s in source_types}
        for score, doc in zip(scores, candidates):
            grouped.setdefault(doc.source_type, []).append((float(score), doc))

        limits = per_source_limits or {}
        out: list[SourceDocument] = []
        for source_type in source_types:
            ranked = sorted(grouped.get(source_type, []), key=lambda x: x[0], reverse=True)
            source_limit = min(max_per_source, max(0, limits.get(source_type, max_per_source)))
            for rank, (_, doc) in enumerate(ranked[:source_limit], start=1):
                clone = doc.model_copy(deep=True)
                clone.retrieval_query = query
                clone.rank = rank
                out.append(clone)
        return out
