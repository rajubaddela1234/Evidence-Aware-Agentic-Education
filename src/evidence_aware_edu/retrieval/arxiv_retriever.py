from __future__ import annotations

from pathlib import Path

import arxiv
import certifi
import pymupdf
import requests

from evidence_aware_edu.retrieval.base import Retriever
from evidence_aware_edu.schemas import SourceDocument
from evidence_aware_edu.utils import stable_id


class ArxivRetriever(Retriever):
    """Search arXiv and extract text from a small number of full papers."""

    source_type = "academic"

    def __init__(
        self,
        *,
        max_papers_per_query: int = 2,
        cache_dir: Path = Path("data/arxiv_text_cache"),
        pdf_timeout_seconds: float = 30.0,
        fallback_to_abstract: bool = True,
    ) -> None:
        self.max_papers_per_query = max(1, int(max_papers_per_query))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_timeout_seconds = float(pdf_timeout_seconds)
        self.fallback_to_abstract = bool(fallback_to_abstract)
        self.client = arxiv.Client(page_size=max(5, self.max_papers_per_query), delay_seconds=3.0, num_retries=3)

    @staticmethod
    def _normalise_text(text: str) -> str:
        # Clean the PDF text but keep paragraph breaks.
        text = (text or "").replace("\x00", "")
        lines = [line.rstrip() for line in text.splitlines()]
        compact: list[str] = []
        blank = False
        for line in lines:
            if line.strip():
                compact.append(line)
                blank = False
            elif not blank:
                compact.append("")
                blank = True
        return "\n".join(compact).strip()

    @classmethod
    def extract_pdf_text(cls, pdf_bytes: bytes) -> tuple[str, int]:
        """Extract text from every page in the PDF."""
        if not pdf_bytes:
            return "", 0
        # MuPDF sometimes prints recoverable structure errors, such as missing
        # optional layer configuration. Those should not interrupt text retrieval.
        # Extraction exceptions still propagate to the abstract fallback.
        show_errors = pymupdf.TOOLS.mupdf_display_errors()
        show_warnings = pymupdf.TOOLS.mupdf_display_warnings()
        pymupdf.TOOLS.mupdf_display_errors(False)
        pymupdf.TOOLS.mupdf_display_warnings(False)
        doc = None
        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            pages: list[str] = []
            for page_no, page in enumerate(doc, start=1):
                text = cls._normalise_text(page.get_text("text"))
                if text:
                    pages.append(f"[Page {page_no}]\n{text}")
            return "\n\n".join(pages).strip(), len(doc)
        finally:
            if doc is not None:
                doc.close()
            pymupdf.TOOLS.mupdf_display_errors(show_errors)
            pymupdf.TOOLS.mupdf_display_warnings(show_warnings)

    def _cache_path(self, entry_id: str) -> Path:
        return self.cache_dir / f"{stable_id('ARXIVTXT', entry_id)}.txt"

    def _load_cached_text(self, entry_id: str) -> str | None:
        path = self._cache_path(entry_id)
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        return text or None

    def _save_cached_text(self, entry_id: str, text: str) -> None:
        path = self._cache_path(entry_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _download_pdf(self, pdf_url: str) -> bytes:
        # Use certifi directly instead of inheriting a machine-specific CA path.
        # This avoids a common Windows failure where REQUESTS_CA_BUNDLE or
        # SSL_CERT_FILE points at an invalid certificate file.
        response = requests.get(
            pdf_url,
            timeout=(10.0, self.pdf_timeout_seconds),
            verify=certifi.where(),
            headers={
                "User-Agent": (
                    "Evidence-Aware-Agentic-Education/1.0 "
                    "(research text extraction; contact via project repository)"
                )
            },
        )
        response.raise_for_status()
        return response.content

    def _paper_text(self, paper) -> tuple[str, dict[str, object]]:
        entry_id = str(paper.entry_id)
        cached = self._load_cached_text(entry_id)
        if cached:
            # The cache keeps text only, so count the saved page markers.
            pages = cached.count("[Page ")
            return cached, {
                "content_mode": "full_pdf_text",
                "full_text_available": True,
                "full_text_cached": True,
                "pages_extracted": pages,
                "text_characters": len(cached),
            }

        pdf_url = str(paper.pdf_url or "").strip()
        if pdf_url:
            try:
                pdf_bytes = self._download_pdf(pdf_url)
                text, pages = self.extract_pdf_text(pdf_bytes)
                if text:
                    self._save_cached_text(entry_id, text)
                    return text, {
                        "content_mode": "full_pdf_text",
                        "full_text_available": True,
                        "full_text_cached": False,
                        "pages_extracted": pages,
                        "text_characters": len(text),
                    }
                full_text_error = "PDF contained no extractable machine-readable text."
            except Exception as exc:
                full_text_error = repr(exc)
        else:
            full_text_error = "No PDF URL was supplied by arXiv."

        abstract = (paper.summary or "").strip()
        if self.fallback_to_abstract and abstract:
            return abstract, {
                "content_mode": "abstract_fallback",
                "full_text_available": False,
                "full_text_cached": False,
                "pages_extracted": 0,
                "text_characters": len(abstract),
                "full_text_error": full_text_error,
            }

        return "", {
            "content_mode": "unavailable",
            "full_text_available": False,
            "full_text_cached": False,
            "pages_extracted": 0,
            "text_characters": 0,
            "full_text_error": full_text_error,
        }

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        # Do not let callers exceed the configured paper limit.
        limit = min(max(1, int(max_results)), self.max_papers_per_query)
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        docs: list[SourceDocument] = []
        for rank, paper in enumerate(self.client.results(search), start=1):
            if rank > limit:
                break
            text, full_text_meta = self._paper_text(paper)
            if not text:
                continue
            url = str(paper.entry_id)
            docs.append(
                SourceDocument(
                    source_id=stable_id("S", "academic", url, query),
                    source_type="academic",
                    title=paper.title.strip(),
                    url=url,
                    text=text,
                    retrieval_query=query,
                    rank=rank,
                    author=", ".join(str(a) for a in paper.authors[:8]),
                    published_at=paper.published.isoformat() if paper.published else None,
                    metadata={
                        "updated": paper.updated.isoformat() if paper.updated else None,
                        "primary_category": paper.primary_category,
                        "pdf_url": paper.pdf_url,
                        "abstract": (paper.summary or "").strip(),
                        "academic_paper_limit_per_query": self.max_papers_per_query,
                        "text_only_extraction": True,
                        "ocr_used": False,
                        **full_text_meta,
                    },
                )
            )
        return docs
