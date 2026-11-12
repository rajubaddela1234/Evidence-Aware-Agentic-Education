"""Relevance filtering for reference pools.

Retrieval tools match on keywords, so a pool can contain unrelated pages or papers (for
example a calculator page in an HTTPS pool). This module asks the LLM to score each retrieved
document against the question and drops the ones that are not about it. It also proposes a few
extra web queries when too few relevant documents survive.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from evidence_aware_edu.schemas import SourceDocument
from evidence_aware_edu.utils import clean_web_text

MIN_RELEVANCE = 5  # 0-10 scale; documents scoring below this are dropped
MIN_DOCUMENTS = 5  # below this many survivors, run extra web queries
EXCERPT_CHARS = 1200
BATCH_SIZE = 12

SCORE_SYSTEM = """You judge whether retrieved documents are useful reference material for answering
a USER QUESTION. Score each document from 0 to 10:
- 9-10: directly explains the topic the question asks about.
- 6-8: relevant background, mechanism, factors or limitations of that topic.
- 3-5: loosely related, or only touches the topic in passing.
- 0-2: unrelated, a different subject that shares a keyword, or boilerplate.
Judge only by the title and excerpt. Return one score per document index, structured output only."""

QUERY_SYSTEM = """You write web search queries that find explanatory pages for a USER QUESTION.
Return exactly three short, distinct queries (at most 12 words each) that would surface
educational or reference pages about the topic. Use plain topic terms, not the question wording.
Structured output only."""


class DocScore(BaseModel):
    index: int
    score: float = Field(ge=0, le=10)


class DocScores(BaseModel):
    scores: list[DocScore] = Field(default_factory=list)


class ExtraQueries(BaseModel):
    queries: list[str] = Field(default_factory=list)


def _describe(index: int, doc: SourceDocument) -> str:
    text = clean_web_text(doc.text) if doc.source_type == "web" else (doc.text or "")
    excerpt = " ".join(text[:EXCERPT_CHARS].split())
    return f"[{index}] ({doc.source_type}) {doc.title}\n{excerpt}"


def score_documents(
    llm: Any, question: str, docs: list[SourceDocument], record: Any = None
) -> list[float]:
    """Return one 0-10 relevance score per document, in order. Unscored documents get 0."""
    scores = [0.0] * len(docs)
    for start in range(0, len(docs), BATCH_SIZE):
        batch = docs[start : start + BATCH_SIZE]
        body = "\n\n".join(_describe(i, d) for i, d in enumerate(batch))
        result, log = llm.generate_structured(
            task="reference_pool_relevance",
            system_instruction=SCORE_SYSTEM,
            prompt=f"USER QUESTION:\n{question}\n\nDOCUMENTS:\n{body}",
            schema=DocScores,
            max_output_tokens=800,
        )
        if record:
            record(log)
        for item in result.scores:
            if 0 <= item.index < len(batch):
                scores[start + item.index] = float(item.score)
    return scores


def filter_relevant(
    llm: Any,
    question: str,
    docs: list[SourceDocument],
    record: Any = None,
    threshold: float = MIN_RELEVANCE,
) -> tuple[list[SourceDocument], list[dict[str, Any]]]:
    """Keep documents scoring at least ``threshold``. Also return a per-document audit list."""
    scores = score_documents(llm, question, docs, record)
    kept: list[SourceDocument] = []
    audit: list[dict[str, Any]] = []
    for doc, score in zip(docs, scores):
        keep = score >= threshold
        audit.append(
            {
                "title": doc.title,
                "url": doc.url,
                "source_type": doc.source_type,
                "relevance": score,
                "kept": keep,
            }
        )
        if keep:
            kept.append(doc)
    return kept, audit


def extra_queries(llm: Any, question: str, record: Any = None) -> list[str]:
    result, log = llm.generate_structured(
        task="reference_pool_extra_queries",
        system_instruction=QUERY_SYSTEM,
        prompt=f"USER QUESTION:\n{question}",
        schema=ExtraQueries,
        max_output_tokens=300,
    )
    if record:
        record(log)
    return [q.strip() for q in result.queries if q.strip()][:3]
