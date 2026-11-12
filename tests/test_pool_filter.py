from evidence_aware_edu.experiments.pool_filter import (
    DocScore,
    DocScores,
    ExtraQueries,
    extra_queries,
    filter_relevant,
)
from evidence_aware_edu.schemas import SourceDocument


class FakeLLM:
    def generate_structured(self, *, task, schema, **_):
        if schema is DocScores:
            return DocScores(scores=[DocScore(index=0, score=9), DocScore(index=1, score=1)]), None
        return ExtraQueries(queries=["a", " b ", "c", "d"]), None


def _doc(title: str) -> SourceDocument:
    return SourceDocument(
        source_type="web", title=title, url=f"https://x/{title}", text="t", retrieval_query="q"
    )


def test_filter_keeps_only_relevant_documents_and_records_audit():
    kept, audit = filter_relevant(FakeLLM(), "q", [_doc("good"), _doc("junk")])
    assert [d.title for d in kept] == ["good"]
    assert [a["kept"] for a in audit] == [True, False]


def test_unscored_documents_are_dropped():
    kept, _ = filter_relevant(FakeLLM(), "q", [_doc("a"), _doc("b"), _doc("c")])
    assert [d.title for d in kept] == ["a"]


def test_extra_queries_are_trimmed_and_capped():
    assert extra_queries(FakeLLM(), "q") == ["a", "b", "c"]
