from __future__ import annotations

from evidence_aware_edu.config import Settings


def make_generation_client(settings: Settings):
    # All six systems use the same Gemini generator.
    if settings.generation_provider != "gemini":
        raise ValueError("Only Gemini generation is supported in this project.")
    settings.validate_generation()
    from evidence_aware_edu.llm.gemini import GeminiClient

    return GeminiClient(settings)
