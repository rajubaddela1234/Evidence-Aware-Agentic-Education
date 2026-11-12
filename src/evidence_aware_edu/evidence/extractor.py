from __future__ import annotations

import re
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from evidence_aware_edu.schemas import Evidence, SourceDocument
from evidence_aware_edu.utils import stable_id


def _chunks(text: str, chunk_chars: int = 1200, overlap_chars: int = 180) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunk = text[start:end]
        if end < len(text):
            # End on a sentence boundary when possible.
            cut = max(chunk.rfind(". "), chunk.rfind("? "), chunk.rfind("! "))
            if cut > chunk_chars * 0.55:
                end = start + cut + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_chars)
    return chunks


def _document_passages(doc: SourceDocument) -> list[str]:
    chunks = _chunks(doc.text)
    if doc.source_type == "academic":
        # Search all chunks for academic papers so later pages are not missed.
        return chunks
    # Web and video text keep a small chunk cap.
    return chunks[:20]


def extract_ranked_evidence(
    query: str,
    documents: Iterable[SourceDocument],
    max_total: int = 18,
    per_document: int = 2,
) -> list[Evidence]:
    candidates: list[tuple[SourceDocument, str]] = []
    for doc in documents:
        for passage in _document_passages(doc):
            candidates.append((doc, passage))
    if not candidates:
        return []

    texts = [query] + [f"{doc.title} {passage}" for doc, passage in candidates]
    try:
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(texts)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    except ValueError:
        scores = [0.0] * len(candidates)

    ranked = sorted(
        [(float(score), doc, passage) for score, (doc, passage) in zip(scores, candidates)],
        key=lambda x: x[0],
        reverse=True,
    )
    per_doc_count: dict[str, int] = {}
    evidence: list[Evidence] = []
    for score, doc, passage in ranked:
        if per_doc_count.get(doc.source_id, 0) >= per_document:
            continue
        eid = stable_id("E", doc.source_id, passage[:240])
        evidence.append(
            Evidence(
                evidence_id=eid,
                source_id=doc.source_id,
                source_type=doc.source_type,
                title=doc.title,
                url=doc.url,
                passage=passage,
                retrieval_query=query,
                relevance_score=score,
                published_at=doc.published_at,
            )
        )
        per_doc_count[doc.source_id] = per_doc_count.get(doc.source_id, 0) + 1
        if len(evidence) >= max_total:
            break
    return evidence
