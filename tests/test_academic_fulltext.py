from __future__ import annotations

import sys
import types

import fitz

# Keep the PDF extraction test runnable even when arxiv is not installed locally.
if "arxiv" not in sys.modules:
    fake_arxiv = types.ModuleType("arxiv")
    fake_arxiv.Client = object
    fake_arxiv.Search = object
    fake_arxiv.SortCriterion = types.SimpleNamespace(Relevance="relevance")
    sys.modules["arxiv"] = fake_arxiv

from evidence_aware_edu.evidence.extractor import extract_ranked_evidence
from evidence_aware_edu.retrieval.arxiv_retriever import ArxivRetriever
from evidence_aware_edu.retrieval.frozen import FrozenCorpusHub
from evidence_aware_edu.schemas import SourceDocument


def test_pdf_text_extraction_reads_every_page_without_ocr() -> None:
    pdf = fitz.open()
    page1 = pdf.new_page()
    page1.insert_text((72, 72), "First page academic evidence")
    page2 = pdf.new_page()
    page2.insert_text((72, 72), "Second page electron flow evidence")
    raw = pdf.tobytes()
    pdf.close()

    text, pages = ArxivRetriever.extract_pdf_text(raw)

    assert pages == 2
    assert "[Page 1]" in text
    assert "First page academic evidence" in text
    assert "[Page 2]" in text
    assert "Second page electron flow evidence" in text


def test_academic_evidence_searches_beyond_first_twenty_chunks() -> None:
    # Put the useful sentence near the end of the paper.
    filler_block = ("background material unrelated to the query. " * 35) + "\n"
    text = filler_block * 28
    text += (
        "During battery discharge electrons flow through the external circuit "
        "while lithium ions move through the electrolyte."
    )
    doc = SourceDocument(
        source_id="S-academic-test",
        source_type="academic",
        title="Full paper test",
        url="https://arxiv.org/abs/test",
        text=text,
        retrieval_query="battery",
    )

    evidence = extract_ranked_evidence(
        "electron flow through the external circuit",
        [doc],
        max_total=1,
        per_document=1,
    )

    assert evidence
    assert "external circuit" in evidence[0].passage.lower()


def test_frozen_retrieval_caps_academic_results_at_two() -> None:
    documents = [
        SourceDocument(
            source_id=f"S-{i}",
            source_type="academic",
            title=f"Battery paper {i}",
            url=f"https://arxiv.org/abs/{i}",
            text="lithium ion battery electron energy storage " * 20,
            retrieval_query="battery",
        )
        for i in range(5)
    ]

    ranked = FrozenCorpusHub.rank(
        "lithium ion battery",
        documents,
        ["academic"],
        max_per_source=5,
        per_source_limits={"academic": 2},
    )

    assert len(ranked) == 2
    assert all(d.source_type == "academic" for d in ranked)
