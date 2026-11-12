from __future__ import annotations

from pathlib import Path

import requests

from evidence_aware_edu.experiments.io import load_benchmark
from evidence_aware_edu.retrieval.youtube import _TranscriptSession, YouTubeRetriever


def test_timeout_session_adds_connect_and_read_timeouts(monkeypatch):
    seen = {}

    def fake_request(self, method, url, **kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(requests.Session, "request", fake_request)
    session = _TranscriptSession(9.0)
    session.request("GET", "https://www.youtube.com/watch?v=abc")
    assert seen["timeout"] == (5.0, 9.0)


def test_timeout_session_respects_explicit_timeout(monkeypatch):
    seen = {}

    def fake_request(self, method, url, **kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(requests.Session, "request", fake_request)
    session = _TranscriptSession(9.0)
    session.request("GET", "https://www.youtube.com", timeout=3)
    assert seen["timeout"] == 3


def test_transcript_failure_does_not_abort_other_sources(capsys):
    retriever = YouTubeRetriever.__new__(YouTubeRetriever)

    class TimedOutTranscript:
        def fetch(self, *args, **kwargs):
            raise requests.exceptions.ReadTimeout("youtube did not respond")

    retriever.transcripts = TimedOutTranscript()
    assert retriever._transcript("a-vid") == ""
    assert capsys.readouterr().out == ""


def test_final_and_pilot_questions_keep_their_ids():
    final = load_benchmark(Path("data/final_benchmark_10.jsonl"))
    pilot = load_benchmark(Path("data/pilot_benchmark_10.jsonl"))
    assert len(final) == 10 and len(pilot) == 10
    assert final[0].prompt_id == "FQ01"
    assert final[0].question == (
        "How does retrieval-augmented generation improve LLM answers, and what can go wrong?"
    )
    assert pilot[0].prompt_id == "Q01"
    assert pilot[0].question == (
        "How does self-attention help transformer models process language, and what limitations does it have?"
    )
    assert len({p.question for p in final}) == 10
    assert len({p.question for p in pilot}) == 10


def test_single_env_file_and_readme():
    root = Path(".")
    assert (root / ".env").is_file()
    assert not (root / ".env.example").exists()
    assert [
        p.name for p in root.rglob("*.md")
        if not any(part in {".pytest_cache", ".venv", "__pycache__"} for part in p.parts)
    ] == ["README.md"]


def test_ip_blocked_stops_further_transcript_attempts(monkeypatch, capsys):
    retriever = YouTubeRetriever.__new__(YouTubeRetriever)
    monkeypatch.setattr(YouTubeRetriever, "_blocked_for_process", False)
    attempted = []

    class IpBlocked(Exception):
        pass

    class BlockedTranscript:
        def fetch(self, video_id, **kwargs):
            attempted.append(video_id)
            raise IpBlocked("blocked by YouTube")

    retriever.transcripts = BlockedTranscript()
    assert retriever._transcript("first") == ""
    assert retriever.ip_blocked
    assert retriever._transcript("second") == ""
    assert attempted == ["first"]
    assert capsys.readouterr().out == ""


def test_ip_blocked_skips_video_search_but_keeps_other_sources_available(monkeypatch, capsys):
    monkeypatch.setattr(YouTubeRetriever, "_blocked_for_process", True)
    retriever = YouTubeRetriever.__new__(YouTubeRetriever)

    class UnexpectedYouTubeCall:
        def search(self):
            raise AssertionError("blocked YouTube should not be queried again")

    retriever.youtube = UnexpectedYouTubeCall()
    assert retriever.search("any topic") == []
    assert capsys.readouterr().out == ""


def test_disabled_transcripts_do_not_print_warnings(monkeypatch, capsys):
    monkeypatch.setattr(YouTubeRetriever, "_blocked_for_process", False)
    retriever = YouTubeRetriever.__new__(YouTubeRetriever)

    class TranscriptsDisabled(Exception):
        pass

    class DisabledTranscript:
        def fetch(self, video_id, **kwargs):
            raise TranscriptsDisabled("no transcript")

    retriever.transcripts = DisabledTranscript()
    assert retriever._transcript("video") == ""
    assert not retriever.ip_blocked
    assert capsys.readouterr().out == ""
