from __future__ import annotations

from tavily import TavilyClient

from evidence_aware_edu.retrieval.base import Retriever
from evidence_aware_edu.schemas import SourceDocument
from evidence_aware_edu.utils import clip, stable_id


class TavilyRetriever(Retriever):
    source_type = "web"

    def __init__(self, api_key: str) -> None:
        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        response = self.client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_raw_content=True,
        )
        docs: list[SourceDocument] = []
        for rank, item in enumerate(response.get("results", []), start=1):
            url = item.get("url") or ""
            title = item.get("title") or url or "Untitled web result"
            text = item.get("raw_content") or item.get("content") or ""
            if not text.strip():
                continue
            docs.append(
                SourceDocument(
                    source_id=stable_id("S", "web", url, query),
                    source_type="web",
                    title=clip(title, 300),
                    url=url,
                    text=text,
                    retrieval_query=query,
                    rank=rank,
                    published_at=item.get("published_date"),
                    metadata={"score": item.get("score")},
                )
            )
        return docs
