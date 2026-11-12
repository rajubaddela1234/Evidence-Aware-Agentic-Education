from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from evidence_aware_edu.agents.claim_extractor import ClaimExtractorAgent
from evidence_aware_edu.agents.verifier import VerifierAgent
from evidence_aware_edu.config import get_settings
from evidence_aware_edu.evaluation.judge import QualityJudge
from evidence_aware_edu.utils import strip_evidence_markers
from evidence_aware_edu.evaluation.metrics import (
    narration_quality_percent,
    query_coverage_percent,
    reference_evidence_support_percent,
    support_counts,
)
from evidence_aware_edu.llm.claude import ClaudeEvaluatorClient
from evidence_aware_edu.media.did import create_did_video
from evidence_aware_edu.observability import configure_langsmith
from evidence_aware_edu.pipelines.factory import PIPELINES, make_pipeline
from evidence_aware_edu.schemas import BenchmarkPrompt
from evidence_aware_edu.telemetry import TelemetryCollector


TITLE = (
    "Evidence-Aware Agentic Educational Content Generation: Adaptive Retrieval, "
    "Claim-Level Verification, and Reliable Video Synthesis"
)


@st.cache_resource
def _settings():
    # The app uses live retrieval; benchmark runs use the saved reference pools.
    configured = get_settings().model_copy(update={"use_frozen_reference_pool": False})
    configure_langsmith(configured)
    return configured


def _evaluate_demo(*, question: str, result, settings):
    """Run the independent evaluator once and keep the result in session state."""
    eval_llm = ClaudeEvaluatorClient(settings, model=settings.evaluator_model)
    eval_telemetry = TelemetryCollector()
    extractor = ClaimExtractorAgent(eval_llm, eval_telemetry)
    verifier = VerifierAgent(eval_llm, eval_telemetry)
    quality_judge = QualityJudge(eval_llm)

    claims = extractor.from_script(result.script)
    if result.evidence:
        verifications = verifier.run(
            claims,
            result.evidence,
            task="demo_evidence_support",
        )
        support_pct = reference_evidence_support_percent(verifications)
        counts = support_counts(verifications)
    else:
        support_pct = None
        counts = None

    quality, quality_log = quality_judge.run(
        question, [], strip_evidence_markers(result.script), record=eval_telemetry.add_llm
    )
    eval_telemetry.add_llm(quality_log)

    return {
        "quality": quality,
        "support_pct": support_pct,
        "counts": counts,
        "telemetry": eval_telemetry.finish(),
    }


def _video_output_path(*, system_id: str, script: str) -> Path:
    script_hash = hashlib.sha256(script.encode("utf-8")).hexdigest()[:12]
    return Path("outputs") / "demo_videos" / f"{system_id}_{script_hash}.mp4"


def _render_video_section(*, result, run_system_id: str, settings) -> None:
    st.subheader("Optional video generation")
    st.caption(
        "Video rendering is manual. It starts only when you press the Generate video button; "
        "running the educational system does not create a video automatically."
    )

    api_key = (settings.did_api_key or "").strip()
    presenter_url = (settings.did_presenter_url or "").strip()
    video_configured = bool(api_key and presenter_url)

    if not video_configured:
        st.info(
            "To enable video rendering, set DID_API_KEY and DID_PRESENTER_URL in .env, "
            "then restart Streamlit."
        )

    generate_clicked = st.button(
        "Generate video",
        key="generate_video_button",
        disabled=not video_configured,
    )

    if generate_clicked:
        output_path = _video_output_path(system_id=run_system_id, script=result.script)
        try:
            with st.spinner("Generating video with D-ID..."):
                rendered = create_did_video(
                    script=result.script,
                    api_key=api_key,
                    presenter_url=presenter_url,
                    voice_id=settings.did_voice_id,
                    output_path=output_path,
                )
        except Exception as exc:
            st.session_state.demo_video_path = None
            st.error("Video generation failed.")
            st.exception(exc)
        else:
            st.session_state.demo_video_path = str(rendered)
            st.success("Video generated successfully.")

    saved_video = st.session_state.get("demo_video_path")
    if saved_video:
        video_path = Path(saved_video)
        if video_path.exists():
            st.video(str(video_path))
            with video_path.open("rb") as video_file:
                st.download_button(
                    "Download video",
                    data=video_file.read(),
                    file_name=video_path.name,
                    mime="video/mp4",
                    key="download_generated_video",
                )


def _render_result(
    *,
    result,
    run_query: str,
    run_system_id: str,
    run_generation_provider: str,
    run_generation_model: str,
    evaluation,
    settings,
) -> None:
    st.caption(
        f"Showing last completed run: `{run_system_id}` · "
        f"generator `{run_generation_provider}` / `{run_generation_model}` · {run_query}"
    )

    st.subheader("Generated script")
    st.write(result.script)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Sources", len(result.sources))
    c2.metric("Evidence", len(result.evidence))
    c3.metric("LLM calls", result.telemetry.total_llm_calls)
    c4.metric("Retrieval calls", result.telemetry.retrieval_calls)
    c5.metric("Latency (s)", f"{result.telemetry.wall_seconds:.1f}")
    c6.metric("Est. generation cost", f"${result.telemetry.estimated_llm_cost_usd:.5f}")

    p1, p2 = st.columns(2)
    p1.metric("LLM-call latency P50", f"{result.telemetry.llm_call_latency_p50_seconds:.2f}s")
    p2.metric("LLM-call latency P90", f"{result.telemetry.llm_call_latency_p90_seconds:.2f}s")

    if evaluation is not None:
        quality = evaluation["quality"]
        support_pct = evaluation["support_pct"]
        counts = evaluation["counts"]
        demo_eval_telemetry = evaluation["telemetry"]

        st.subheader("Independent-provider quality evaluation (Claude)")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Query Coverage", f"{query_coverage_percent(quality):.1f}%")
        q2.metric(
            "Run Evidence Support",
            f"{support_pct:.1f}%" if support_pct is not None else "N/A",
        )
        q3.metric("Query Relevance", f"{quality.relevance_score * 100:.1f}%")
        q4.metric("Narration Quality", f"{narration_quality_percent(quality):.1f}%")

        st.caption(
            "Evidence Support measures agreement with the evidence supplied to the evaluator. "
            "It is not a factual-accuracy or truth score. In the controlled benchmark every system "
            "is evaluated against the same frozen question-specific reference corpus."
        )

        if quality.aspect_scores:
            st.caption("Coverage by question requirement")
            st.dataframe(
                [
                    {
                        "Aspect": x.aspect,
                        "Coverage %": round(x.score * 100, 1),
                        "Reason": x.rationale,
                    }
                    for x in quality.aspect_scores
                ],
                use_container_width=True,
            )

        st.caption(
            "Narration dimensions (0–4 each): "
            f"opening {quality.opening_orientation_score}, "
            f"introduction/roadmap {quality.introduction_roadmap_score}, "
            f"definitions/foundations {quality.definitions_foundations_score}, "
            f"explanatory depth {quality.explanatory_depth_score}, "
            f"examples/applications {quality.examples_applications_score}, "
            f"organization/transitions {quality.organization_transitions_score}, "
            f"clarity {quality.clarity_accessibility_score}, "
            f"spoken naturalness {quality.spoken_naturalness_score}, "
            f"conclusion {quality.conclusion_takeaway_score}, "
            f"reflective ending {quality.reflective_ending_score}."
        )
        if counts is not None:
            st.caption(
                "Evidence-support labels — "
                f"supported: {counts['supported']}, partial: {counts['partial']}, "
                f"unsupported: {counts['unsupported']}, contradicted: {counts['contradicted']}"
            )
        st.caption(
            f"Evaluation overhead: {demo_eval_telemetry.total_llm_calls} evaluator calls, "
            f"${demo_eval_telemetry.estimated_llm_cost_usd:.5f} Claude CLI list-price-equivalent "
            "telemetry. Subscription billing and allowance usage can differ from this value."
        )

    if run_system_id == "P1" and result.metadata:
        st.subheader("Coverage-aware retrieval audit")
        plan = result.metadata.get("research_plan") or {}
        completeness = result.metadata.get("requirement_completeness") or {}
        if completeness:
            st.write("**Requirement completeness check**")
            st.json(completeness)
        if plan.get("answer_requirements"):
            st.write("**Final answer requirements after completeness checking**")
            st.write(plan["answer_requirements"])
        if result.metadata.get("requirement_assessments"):
            st.write("**Final requirement sufficiency**")
            st.dataframe(result.metadata["requirement_assessments"], use_container_width=True)
        if result.metadata.get("retrieval_events"):
            st.write("**Adaptive retrieval events**")
            st.dataframe(result.metadata["retrieval_events"], use_container_width=True)
        if result.contradictions:
            st.write("**Evidence disagreements affecting generation**")
            st.dataframe([c.model_dump() for c in result.contradictions], use_container_width=True)
        if result.metadata.get("final_requirement_audit"):
            st.write("**Final narration requirement audit**")
            st.dataframe(result.metadata["final_requirement_audit"], use_container_width=True)
            st.caption(
                f"Final repair performed: {bool(result.metadata.get('repair_performed', False))}"
            )

    if result.telemetry.llm_calls:
        st.subheader("LLM execution telemetry")
        st.dataframe(
            [
                {
                    "No.": i,
                    "Task": call.task,
                    "Model": call.model,
                    "Latency (s)": round(call.latency_seconds, 2),
                    "Prompt tokens": call.prompt_tokens,
                    "Output tokens": call.output_tokens,
                    "Thinking tokens": call.thinking_tokens,
                    "Billable output": call.output_tokens + call.thinking_tokens,
                    "Total tokens": call.total_tokens,
                    "Cost (USD)": round(call.total_cost_usd, 6),
                    "Attempts": call.attempts,
                }
                for i, call in enumerate(result.telemetry.llm_calls, start=1)
            ],
            use_container_width=True,
        )

    if result.verifications:
        st.subheader("Internal claim-evidence verification")
        st.dataframe([v.model_dump() for v in result.verifications], use_container_width=True)

    if result.sources:
        st.subheader("Research sources")
        for source in result.sources:
            with st.expander(f"[{source.source_type}] {source.title}"):
                st.write(source.url)
                if source.source_type == "academic":
                    mode = source.metadata.get("content_mode", "unknown")
                    pages = source.metadata.get("pages_extracted", 0)
                    chars = source.metadata.get("text_characters", len(source.text))
                    st.caption(
                        f"Academic content: {mode} · pages extracted: {pages} · "
                        f"text characters: {chars:,} · OCR: no"
                    )
                st.write(source.text[:2000])

    _render_video_section(
        result=result,
        run_system_id=run_system_id,
        settings=settings,
    )


st.set_page_config(page_title="Evidence-Aware Agentic Education", layout="wide")
st.title(TITLE)
st.caption(
    "Research demonstrator: five fixed baselines plus a coverage-aware, evidence-aware agentic system."
)

settings = _settings()

if "demo_result" not in st.session_state:
    st.session_state.demo_result = None
if "demo_evaluation" not in st.session_state:
    st.session_state.demo_evaluation = None
if "demo_query" not in st.session_state:
    st.session_state.demo_query = None
if "demo_system_id" not in st.session_state:
    st.session_state.demo_system_id = None
if "demo_video_path" not in st.session_state:
    st.session_state.demo_video_path = None
if "demo_generation_provider" not in st.session_state:
    st.session_state.demo_generation_provider = None
if "demo_generation_model" not in st.session_state:
    st.session_state.demo_generation_model = None

with st.sidebar:
    run_settings = settings

    system_id = st.selectbox(
        "Architecture / system",
        list(PIPELINES.keys()),
        index=list(PIPELINES.keys()).index("P1"),
        format_func=lambda x: f"{x} — {PIPELINES[x].system_name}",
    )

    try:
        run_settings.validate_generation()
        generation_ok = True
        generation_error = ""
    except ValueError as exc:
        generation_ok = False
        generation_error = str(exc)

    st.write(
        f"Generation: `{run_settings.generation_provider}` · "
        f"`{run_settings.generation_access}` · `{run_settings.generation_model}`"
    )
    if not generation_ok:
        st.error(generation_error)

    if run_settings.claude_access_method == "api":
        anthropic_key = (run_settings.anthropic_api_key or "").strip()
        evaluator_ok = bool(anthropic_key and not anthropic_key.lower().startswith("your_"))
        evaluator_label = f"Anthropic API / {run_settings.claude_evaluator_model}"
        evaluator_error = "ANTHROPIC_API_KEY is missing for Claude API evaluation."
    else:
        evaluator_ok = ClaudeEvaluatorClient.cli_is_available()
        evaluator_label = f"Claude Code CLI / {run_settings.claude_evaluator_model}"
        evaluator_error = "Claude Code was not found. Install it and sign in before evaluation."

    if evaluator_ok:
        st.write(f"Independent evaluator: `{evaluator_label}`")
    else:
        st.error(evaluator_error)

    reasoning = run_settings.thinking_level
    st.write(f"Generation reasoning setting: `{reasoning}`")
    st.write(
        f"arXiv: up to `{run_settings.arxiv_max_papers_per_query}` full-text papers/query "
        "(text layer only)"
    )
    if run_settings.youtube_require_transcript:
        st.write("YouTube evidence: `transcript-only` (no description fallback)")
    if run_settings.langsmith_tracing and run_settings.langsmith_api_key:
        st.success(f"LangSmith: ON · `{run_settings.langsmith_project}`")
    else:
        st.info("LangSmith: OFF")
    input_rate, output_rate = run_settings.gemini_pricing()
    st.caption(
        f"Configured list-price estimate: ${input_rate:.2f}/1M input, "
        f"${output_rate:.2f}/1M output tokens. Subscription billing can differ."
    )
    st.warning("Demo mode uses live retrieval. The benchmark uses frozen question-specific pools.")

architecture_figure = Path(f"research_assets/figures/Architecture_{system_id}.png")
if architecture_figure.exists():
    st.subheader(f"Selected architecture: {system_id} — {PIPELINES[system_id].system_name}")
    st.image(str(architecture_figure), use_container_width=True)

query = st.text_area(
    "Educational research question",
    value=(
        "How does a transformer use self-attention to determine which tokens matter "
        "when generating the next token?"
    ),
    height=120,
)

run_clicked = st.button(
    "Run system",
    type="primary",
    disabled=(not query.strip()) or (not generation_ok),
)

if run_clicked:
    prompt = BenchmarkPrompt(
        prompt_id="DEMO",
        domain="Demo",
        reasoning_type="demo",
        question=query.strip(),
    )

    with st.spinner(f"Running {system_id}..."):
        try:
            result = make_pipeline(system_id, run_settings).run(prompt)
        except Exception as exc:
            st.exception(exc)
            st.stop()

    evaluation = None
    if evaluator_ok:
        with st.spinner("Evaluating support, coverage, relevance, and narration quality..."):
            try:
                evaluation = _evaluate_demo(
                    question=query.strip(),
                    result=result,
                    settings=run_settings,
                )
            except Exception as exc:
                st.warning("Generation completed, but independent evaluation failed.")
                st.exception(exc)

    st.session_state.demo_result = result
    st.session_state.demo_evaluation = evaluation
    st.session_state.demo_query = query.strip()
    st.session_state.demo_system_id = system_id
    st.session_state.demo_generation_provider = run_settings.generation_provider
    st.session_state.demo_generation_model = run_settings.generation_model
    # A new generated script must never reuse the previous run's video.
    st.session_state.demo_video_path = None

if st.session_state.demo_result is not None:
    _render_result(
        result=st.session_state.demo_result,
        run_query=st.session_state.demo_query,
        run_system_id=st.session_state.demo_system_id,
        run_generation_provider=st.session_state.demo_generation_provider or "unknown",
        run_generation_model=st.session_state.demo_generation_model or "unknown",
        evaluation=st.session_state.demo_evaluation,
        settings=settings,
    )
