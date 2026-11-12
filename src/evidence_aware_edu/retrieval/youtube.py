from __future__ import annotations

import logging

import requests


logger = logging.getLogger(__name__)

from evidence_aware_edu.retrieval.base import Retriever
from evidence_aware_edu.schemas import SourceDocument
from evidence_aware_edu.utils import clip, stable_id


class _TranscriptSession(requests.Session):
    """Supply timeouts to the requests made inside youtube-transcript-api."""

    def __init__(self, read_timeout_seconds: float) -> None:
        super().__init__()
        self.read_timeout_seconds = float(read_timeout_seconds)

    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", (min(5.0, self.read_timeout_seconds), self.read_timeout_seconds))
        return super().request(method, url, **kwargs)


class YouTubeRetriever(Retriever):
    """Use transcripts as video evidence; skip a video if its captions cannot be read."""

    source_type = "video"

    # One blocked transcript request is enough to identify a blocked IP for this
    # Python process. Do not retry the same inaccessible endpoint for every topic.
    _blocked_for_process = False

    @property
    def ip_blocked(self) -> bool:
        return type(self)._blocked_for_process

    def __init__(
        self,
        api_key: str,
        transcript_required: bool = True,
        transcript_timeout_seconds: float = 10.0,
        max_transcript_candidates: int = 8,
    ) -> None:
        try:
            from googleapiclient.discovery import build
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError as exc:
            raise RuntimeError("YouTube retrieval dependencies are missing. Run `uv sync`.") from exc

        self.youtube = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
        self.transcripts = YouTubeTranscriptApi(
            http_client=_TranscriptSession(transcript_timeout_seconds)
        )
        self.transcript_required = transcript_required
        self.max_transcript_candidates = max(1, int(max_transcript_candidates))
        self.transcript_timeout_seconds = transcript_timeout_seconds

    def _transcript(self, video_id: str) -> str:
        if self.ip_blocked:
            return ""
        try:
            fetched = self.transcripts.fetch(video_id, languages=["en"])
            return " ".join(snippet.text for snippet in fetched).strip()
        except requests.exceptions.Timeout:
            # A timeout applies to this video. Other retrieval sources can still run.
            logger.debug("YouTube transcript timed out for %s", video_id)
        except requests.exceptions.RequestException as exc:
            logger.debug("YouTube network request failed for %s: %s", video_id, type(exc).__name__)
        except Exception as exc:
            # youtube-transcript-api returns IpBlocked when YouTube denies transcript
            # access to the current IP. Further attempts in this process are wasteful.
            error_name = type(exc).__name__
            if error_name in {"IpBlocked", "RequestBlocked"}:
                type(self)._blocked_for_process = True
            logger.debug("YouTube transcript unavailable for %s: %s", video_id, error_name)
        return ""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        # Skip the YouTube endpoint if this process has already encountered an IP block.
        if self.ip_blocked:
            return []

        # A hard cap prevents large searches from trying dozens of unavailable videos.
        max_results = max(1, int(max_results))
        candidate_count = min(
            50, max(max_results, min(max_results * 4, getattr(self, "max_transcript_candidates", 8)))
        )
        response = (
            self.youtube.search()
            .list(
                q=query,
                part="snippet",
                type="video",
                maxResults=candidate_count,
                relevanceLanguage="en",
                safeSearch="moderate",
            )
            .execute()
        )

        docs: list[SourceDocument] = []
        for search_rank, item in enumerate(response.get("items", []), start=1):
            if len(docs) >= max_results:
                break
            video_id = item.get("id", {}).get("videoId")
            if not video_id:
                continue
            snippet = item.get("snippet", {})
            transcript = self._transcript(video_id)
            if self.ip_blocked:
                break

            if self.transcript_required and not transcript:
                continue

            # The final benchmark requires transcripts. Description-only text is
            # allowed for the demo only if this check is disabled explicitly.
            description = (snippet.get("description") or "").strip()
            text = transcript if transcript else description
            if not text:
                continue

            docs.append(
                SourceDocument(
                    source_id=stable_id("S", "video", video_id, query),
                    source_type="video",
                    title=clip(snippet.get("title", "Untitled video"), 300),
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    text=text,
                    retrieval_query=query,
                    rank=len(docs) + 1,
                    author=snippet.get("channelTitle"),
                    published_at=snippet.get("publishedAt"),
                    metadata={
                        "video_id": video_id,
                        "has_transcript": bool(transcript),
                        "transcript_required": self.transcript_required,
                        "text_source": "transcript" if transcript else "description",
                        "youtube_search_rank": search_rank,
                    },
                )
            )

        return docs
