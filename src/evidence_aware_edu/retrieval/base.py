from __future__ import annotations

from abc import ABC, abstractmethod

from evidence_aware_edu.schemas import SourceDocument, SourceType


class Retriever(ABC):
    source_type: SourceType

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        raise NotImplementedError
