"""Tests for the measurement, P1 retrieval, statistics and reproducibility improvements."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from evidence_aware_edu.agents.router import RouterAgent
from evidence_aware_edu.config import (
    PILOT_OUTPUT_DIR,
    PILOT_REFERENCE_POOL_DIR,
    use_benchmark_directories,
)
from evidence_aware_edu.evaluation.evaluator import (
    CLAIM_REFERENCE_PASSAGES,
    GENERAL_REFERENCE_PASSAGES,
    ResearchEvaluator,
)
from evidence_aware_edu.evaluation.statistics import (
    bootstrap_ci,
    pairwise_significance,
    system_means_with_ci,
)
from evidence_aware_edu.experiments.identity import (
    EVALUATION_CODE_PATHS,
    GENERATION_CODE_PATHS,
    code_sha256,
)
from evidence_aware_edu.graph.evidence_aware_graph import EvidenceAwareGraph
from evidence_aware_edu.schemas import Claim, RouteDecision, SourceDocument
from evidence_aware_edu.utils import clean_web_text


def doc(source_id: str, text: str, source_type: str = "web") -> SourceDocument:
    return SourceDocument(
        source_id=source_id, source_type=source_type, title=source_id,
        url=f"https://example.org/{source_id}", text=text, retrieval_query="q",
    )


# --- web text cleaning --------------------------------------------------------------------

def test_clean_web_text_drops_menus_images_and_link_lists_but_keeps_prose():
    raw = (
        "[Jump to content](#bodyContent)\n"
        "![Logo](/static/logo.svg)\n"
        "* [1 Overview](#Overview)\n"
        "[Search](/wiki/Special:Search \"Search\")\n"
        "# Retrieval-augmented generation\n"
        "**RAG** lets a [language model](/wiki/LLM) use retrieved documents when it answers.[[1]](#cite-1)\n"
        "Home\n"
    )
    cleaned = clean_web_text(raw)
    assert "Jump to content" not in cleaned
    assert "Logo" not in cleaned and "Search" not in cleaned and "Overview" not in cleaned
    assert "# Retrieval-augmented generation" in cleaned
    assert "lets a language model use retrieved documents when it answers." in cleaned
    assert "cite-1" not in cleaned and "Home" not in cleaned


# --- 1. reference support: claim-specific passages come first ------------------------------

class CapturingVerifier:
    def __init__(self):
        self.batches: list[list[str]] = []

    def run(self, claims, evidence, **_):
        self.batches.append([e.source_id for e in evidence])
        return []


def test_reference_support_puts_claim_passages_before_general_ones():
    docs = [doc(f"d{i}", f"topic{i} " * 30 + f"unique{i} facts about subject{i}.") for i in range(12)]
    claims = [Claim(claim_id="c1", text="unique3 facts about subject3"),
              Claim(claim_id="c2", text="unique7 facts about subject7")]
    evaluator = object.__new__(ResearchEvaluator)
    evaluator.settings = SimpleNamespace(claude_evaluator_max_context_chars=40000)
    verifier = CapturingVerifier()
    _, evidence_map = evaluator._claim_specific_reference_support(
        "a question with no shared words", claims, docs, verifier
    )
    sent = verifier.batches[0]
    # Each claim's best passage is first, interleaved, before any general passage.
    assert sent[:2] == ["d3", "d7"]
    assert len(evidence_map["c1"]) <= CLAIM_REFERENCE_PASSAGES
    assert len(sent) <= 2 * CLAIM_REFERENCE_PASSAGES + GENERAL_REFERENCE_PASSAGES


# --- 3. router triangulates across at least two source types -------------------------------

class FixedRouteLLM:
    def __init__(self, sources):
        self.sources = sources

    def generate_structured(self, **_):
        return RouteDecision(sources=self.sources, max_per_source=2), "log"


def test_router_adds_a_second_source_type_when_model_picks_one():
    telemetry = SimpleNamespace(add_llm=lambda log: None)
    decision = RouterAgent(FixedRouteLLM(["academic"]), telemetry).run("q")
    assert decision.sources == ["academic", "web"]
    decision = RouterAgent(FixedRouteLLM(["video", "web"]), telemetry).run("q")
    assert decision.sources == ["video", "web"]


# --- 4. frozen-mode gap retrieval searches passages across the whole pool ------------------

def test_frozen_gap_retrieval_adds_passages_from_documents_not_yet_retrieved():
    pool = [
        doc("a", "solar panels convert sunlight " * 20),
        doc("b", "inverters convert direct current to alternating current for homes " * 5),
    ]
    graph = object.__new__(EvidenceAwareGraph)
    graph.settings = SimpleNamespace(use_frozen_reference_pool=True)
    graph.prompt = SimpleNamespace(prompt_id="FQ03", question="How do solar panels work?")
    graph.hub = SimpleNamespace(frozen=SimpleNamespace(load=lambda prompt_id: pool))
    graph.telemetry = None
    state = {"sources": [pool[0]], "evidence": []}
    sources, evidence = graph._gap_retrieve(
        state, ["inverter converts direct current to alternating current"], "inverter", cap=10
    )
    assert any(e.source_id == "b" for e in evidence)
    assert {s.source_id for s in sources} == {"a", "b"}


# --- 6. statistics -------------------------------------------------------------------------

def _two_system_frame(n=8):
    rows = []
    for i in range(n):
        rows.append({"system_id": "A", "prompt_id": f"Q{i}", "query_coverage_percent": 80 + i,
                     "latency_seconds": 5.0})
        rows.append({"system_id": "B", "prompt_id": f"Q{i}", "query_coverage_percent": 60 + i,
                     "latency_seconds": 9.0})
    return pd.DataFrame(rows)


def test_pairwise_significance_detects_consistent_winner_in_both_directions():
    out = pairwise_significance(_two_system_frame(), ["query_coverage_percent", "latency_seconds"])
    coverage = out[out.metric == "query_coverage_percent"].iloc[0]
    latency = out[out.metric == "latency_seconds"].iloc[0]
    assert coverage.a_better_count == 8 and coverage.rank_biserial_a_better == 1.0
    assert coverage.p_holm < 0.05 and bool(coverage.significant_0_05)
    # Lower latency is better, so A (5s) beats B (9s).
    assert latency.preferred_direction == "lower" and latency.a_better_count == 8


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    values = np.array([60.0, 70.0, 75.0, 80.0, 90.0])
    low, high = bootstrap_ci(values)
    assert low <= values.mean() <= high
    assert (low, high) == bootstrap_ci(values)
    table = system_means_with_ci(_two_system_frame(), ["query_coverage_percent"])
    assert set(table.system_id) == {"A", "B"} and (table.ci95_low <= table["mean"]).all()


# --- 8. code hashes ------------------------------------------------------------------------

def test_code_hashes_are_stable_and_distinguish_generation_from_evaluation():
    assert code_sha256(GENERATION_CODE_PATHS) == code_sha256(GENERATION_CODE_PATHS)
    assert code_sha256(GENERATION_CODE_PATHS) != code_sha256(EVALUATION_CODE_PATHS)


# --- 9. pilot folders ----------------------------------------------------------------------

def test_pilot_benchmark_uses_its_own_folders(monkeypatch):
    monkeypatch.delenv("REFERENCE_POOL_DIR", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    assert use_benchmark_directories("data/final_benchmark_10.jsonl") is False
    assert "OUTPUT_DIR" not in os.environ
    assert use_benchmark_directories("data/pilot_benchmark_10.jsonl") is True
    assert os.environ["REFERENCE_POOL_DIR"] == PILOT_REFERENCE_POOL_DIR
    assert os.environ["OUTPUT_DIR"] == PILOT_OUTPUT_DIR
