import json
from types import MethodType

from evidence_aware_edu.retrieval.frozen import FrozenCorpusHub
from evidence_aware_edu.retrieval.youtube import YouTubeRetriever


class _SearchRequest:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class _SearchResource:
    def __init__(self, payload):
        self.payload = payload
        self.kwargs = None

    def list(self, **kwargs):
        self.kwargs = kwargs
        return _SearchRequest(self.payload)


class _YouTubeService:
    def __init__(self, payload):
        self.resource = _SearchResource(payload)

    def search(self):
        return self.resource


def _item(video_id: str, title: str):
    return {
        "id": {"videoId": video_id},
        "snippet": {
            "title": title,
            "description": f"description for {title}",
            "channelTitle": "channel",
            "publishedAt": "2026-01-01T00:00:00Z",
        },
    }


def test_youtube_drops_video_without_transcript_and_uses_no_description_fallback():
    retriever = YouTubeRetriever.__new__(YouTubeRetriever)
    retriever.youtube = _YouTubeService(
        {"items": [_item("no-transcript", "No transcript"), _item("good", "Good video")]}
    )
    retriever.transcript_required = True

    def fake_transcript(self, video_id):
        return "spoken transcript text" if video_id == "good" else ""

    retriever._transcript = MethodType(fake_transcript, retriever)
    docs = retriever.search("agentic systems", max_results=1)

    assert len(docs) == 1
    assert docs[0].metadata["video_id"] == "good"
    assert docs[0].text == "spoken transcript text"
    assert docs[0].metadata["has_transcript"] is True
    assert docs[0].metadata["text_source"] == "transcript"
    # The retriever should keep searching until it finds a transcript.
    assert retriever.youtube.resource.kwargs["maxResults"] > 1


def test_frozen_pool_filters_legacy_description_only_videos(tmp_path):
    pool = tmp_path / "Q01.json"
    pool.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "source_id": "S1",
                        "source_type": "video",
                        "title": "legacy description",
                        "url": "https://youtube.test/1",
                        "text": "description only",
                        "retrieval_query": "q",
                        "metadata": {"has_transcript": False},
                    },
                    {
                        "source_id": "S2",
                        "source_type": "video",
                        "title": "transcript video",
                        "url": "https://youtube.test/2",
                        "text": "actual transcript",
                        "retrieval_query": "q",
                        "metadata": {"has_transcript": True},
                    },
                    {
                        "source_id": "S3",
                        "source_type": "web",
                        "title": "web",
                        "url": "https://example.test",
                        "text": "web text",
                        "retrieval_query": "q",
                        "metadata": {},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    docs = FrozenCorpusHub(tmp_path, require_video_transcript=True).load("Q01")
    assert {d.source_id for d in docs} == {"S2", "S3"}
