# Evidence-Aware Agentic Educational Script Generation

Gemini-only generation with independent Claude evaluation. This is the complete source code for six benchmark pipelines (B1–B5 and P1), a ten-question final benchmark plus a separate ten-question pilot, five new automatic research measurements, and PNG research graphics.

**Version:** 1.4.0  
**Python:** 3.12  
**Generator:** `gemini-3.8-flash`  
**Evaluator:** Claude (CLI or Anthropic API)  
**Markdown files:** this `README.md` only  
**Configuration:** `.env` 
**Graphics:** PNG only, including six separate architecture images.

The code creates and scores educational **scripts**, not finished audio/video or student-learning outcomes. The Streamlit demonstration is a separate workflow.

## 0. What changed in v1.6 (evaluation correctness and reproducibility)

Results from earlier versions are **not comparable**; start a new experiment.

**Scoring rubrics**
- **Query coverage uses four levels per aspect:** 0.9–1.0 main idea clearly explained (brief is fine); 0.8–0.9 explained with one part thin or implicit; 0.7–0.8 present but only mentioned, partly explained or vague; 0.0 absent or materially wrong. Values between 0.0 and 0.7 are rejected and re-asked, so any aspect that appears in the script scores at least 0.7. Explanation anywhere in the script, in any wording, counts. The rubric was deliberately calibrated to give generous credit to any aspect the script addresses; report this in your methods, since it compresses differences between partial and full coverage.
- **Narration** keeps the ten 0–4 dimensions but is judged generously: 4 when a dimension is handled well, 3 when present but plain or brief, 2 or below only when clearly missing or hard to follow. A slightly formal tone or a single minor issue does not lower a score, and the judge takes the higher score when unsure.
- The checklist deriver writes six one-idea obligations of at most 15 words and does not add topics the question does not ask about.
- The grounded-requirement / context-recall judge still uses 0 / 0.5 / 1.

**Measurement fixes**
- **Reference support:** each claim's own top-5 pool passages are sent first (interleaved across the batch), followed by 3 question-level passages; `CLAUDE_EVALUATOR_MAX_CONTEXT_CHARS` is 40000. Previously most claim-specific passages were cut off.
- **Citation checking** gives every claim an explicit ALLOWED EVIDENCE list (its own citations) and discards any evidence ID outside it, so one claim can no longer be credited with another claim's citation. Claims with no resolvable citation are unsupported without a model call.
- **Grounded coverage / context precision / context recall** see complete passages (the old 450-character cut is gone); long inputs are batched, never truncated.
- **Incomplete judgements are not zeros.** If Claude omits a checklist aspect, a ranked passage or a claim, it is re-asked; if it still omits it, the case fails (`RubricViolation` / `IncompleteVerification`) and can be resumed, instead of recording a low score.
- **Web pages are cleaned** (menus, images, link lists, footnote links) when a pool is loaded; pool files and their hashes are unchanged.
- Evidence markers are stripped as whole runs (`[E-a], [E-b]`), so the narration judge no longer sees stray commas.

**Integrity**
- All six checklists are derived **before** any case is evaluated and their hashes are stored in the manifest (`checklist_sha256`). A changed or missing checklist stops the run (`ChecklistIntegrityError`) rather than being silently re-derived.
- The manifest records `generation_code_sha256` and `evaluation_code_sha256` (hashes of the prompt-bearing source files). Resuming with changed code is refused.
- Empty Gemini output is retried and output that hits the token limit fails immediately; a blank script is never saved or scored.
- Pilot runs (`--benchmark data/pilot_benchmark_10.jsonl`) use `data/reference_pools_pilot_10q` and `outputs_pilot_10q`, never the final folders.

**Systems and analysis**
- P1's router always uses at least two source types. In frozen mode, P1's gap retrieval ranks passages across the whole pool (previously it re-selected the same documents and added nothing).
- Grounded prompts allow fact-free analogies and ask for every part of the question; P1 plans 5–8 one-idea requirements and extracts 15–30 claims; `MAX_EVIDENCE_PER_QUERY` is 16 and `MAX_SOURCES_PER_TOOL` is 4 (each system may take up to 4 documents per source type from the frozen pool).
- The claim verifier works in batches of 10 claims, so P1/B5 runs with many claims no longer fail on a cut-off JSON answer.
- Analysis also writes `system_means_ci95.csv` (bootstrap 95% CIs) and `pairwise_wilcoxon_holm.csv` (Wilcoxon signed-rank for every system pair, Holm-corrected, rank-biserial effect size).

## 1. What changed in v1.5.0 (evaluation fairness and a 10-question benchmark)

- **Ten final questions.** `data/final_benchmark_10.jsonl` (FQ01-FQ10) replaces `final_benchmark_8.jsonl`. FQ09 (stellar fusion) and FQ10 (rust and corrosion) are new. The pilot now has ten questions (Q01-Q10) (`data/pilot_benchmark_10.jsonl`). Default final run: **10 questions x 6 systems = 60 scripts and 60 evaluations**.
- **No hand-written `expected_aspects`, no `cutoff_date`.** Benchmark rows now contain only `prompt_id`, `domain`, `reasoning_type` and `question`. The retrieval date filter was removed (it was always `null`). Existing frozen pools remain compatible.
- **Shared question-derived checklist.** Coverage, grounded requirement coverage, context recall and requirement preservation still need a fixed denominator. The evaluator therefore asks Claude **once per question** for exactly six answer obligations, derived from the question text alone, saves them to `<output_dir>/experiments/<EXPERIMENT_ID>/checklists/<QUESTION>.json`, and uses that same list for every system. It is never shown to a generator or stored in a reference pool. The derivation cost is recorded in that file and is not charged to any one system's evaluation cost. Because the checklist is written by an LLM, treat it as a judge-side rubric, not a human gold standard.
- **Clean narration input.** `[E-...]` evidence markers are removed before the narration/coverage judges and before video rendering; claim and citation checks still use the marked script.
- **Complete scripts.** The grounded-requirements judge no longer truncates the script at 5,500 characters. Only the evidence is batched when the prompt would exceed `CLAUDE_EVALUATOR_MAX_JUDGE_PROMPT_CHARS` (default 60000).
- **Enforced rubric.** (Superseded in v1.6: query coverage is now continuous 0.0–1.0; the grounded-requirement judge still uses 0 / 0.5 / 1.) Invalid or incomplete judgments are re-requested up to twice, then the case fails with `RubricViolation` instead of recording an off-rubric score.
- **Not comparable with v1.4.0 results.** Scores from earlier experiments used hand-written aspects and marker-containing narration input. Start a new experiment ID and re-score; do not mix them.

## 1b. What changed in v1.4.0

- (Superseded by v1.5.0: the final set is now ten questions.) The former final FQ09 and FQ10 are pilot Q06 and Q07. The new pilot Q08 is about enzymes, temperature and pH. The original pilot Q01–Q05 are unchanged.
- (Superseded by v1.5.0: the default final benchmark is 10 questions x 6 systems = 60 cases.) Running the pilot separately is 48 cases.
- Gemini is now `gemini-3.8-flash`, with default low thinking and the **2026 introductory API cost estimates** of `$0.75 / 1M input tokens` and `$3.75 / 1M output tokens`. These prices may change. Actual billing may differ from estimates.
- Kept every original metric and added **exactly five** automatic measurements: grounded requirement coverage, context precision@5, context recall, repair effectiveness, and requirement preservation rate. These use extra Claude calls only where appropriate.
- Removed the original `Figure_02_proposed_architecture.png` and the function that wrote it. Added `Architecture_B1.png` through `Architecture_P1.png`, a separate architectural diagram for each implemented system.
- Kept analysis figures and added four measured-result plots, Figures 21–24. The new figures are not populated with invented experimental results.
- Uses new directories `data/reference_pools_final_10q` and `outputs_final_10q`, deliberately isolated from earlier studies. Start a **new experiment ID** for this revision. Do not combine 3.5 Flash-Lite and 3.8 Flash results in a single experiment.
- Kept quiet YouTube error handling, the arXiv text/abstract fallback and frozen-pool compatibility checks from v1.3.5.

## 2. Six systems

| System | What generation actually does |
|---|---|
| B1 | Gemini responds directly without generation-side retrieval |
| B2 | Frozen single-source web retrieval → ranked excerpts → Gemini synthesis |
| B3 | Frozen web, academic and transcript-backed video retrieval → evidence ranking → Gemini synthesis |
| B4 | Decomposition, retrieval, synthesis, feedback, optional query revision/another iteration |
| B5 | B4, followed by claim extraction, evidence verification, whole-script repair and re-verification if changed |
| P1 | Requirement-aware planning, adaptive retrieval, evidence audits, targeted retrieval, claim checks, pedagogy/narration, final audit and conditional repair |

All systems use the same generation model and fixed per-question corpus in a given experiment. The independent Claude evaluator runs after saving each script.

## 3. Setup on Windows Command Prompt (CMD)

Extract the ZIP. In File Explorer open the extracted `evidence_aware_agentic_education` folder (the one with `.env`, `pyproject.toml` and `scripts`). Click the Explorer address bar, type `cmd` and press Enter.

Install Python 3.12 and the dependencies:

```cmd
uv python install 3.12
uv sync --group dev
```

If `uv` is not installed, run `winget install --id astral-sh.uv -e`, close the CMD window and reopen it in the project folder.

Edit the single configuration file:

```cmd
notepad .env
```

Enter your own `GEMINI_API_KEY`, `TAVILY_API_KEY` and `YOUTUBE_API_KEY`. The YouTube transcript provider may block transcripts, in which case academic and web evidence can still be collected. No API keys are supplied in the ZIP. Never upload or commit your filled `.env`.

The generator should look like:

```dotenv
GEMINI_MODEL=gemini-3.8-flash
GEMINI_THINKING_LEVEL=low
GEMINI_INPUT_USD_PER_1M=0.75
GEMINI_OUTPUT_USD_PER_1M=3.75
```

For evaluation choose either authenticated local Claude Code (`CLAUDE_ACCESS_METHOD=cli`) or Anthropic API (`CLAUDE_ACCESS_METHOD=api` with your `ANTHROPIC_API_KEY`). Check CLI access with:

```cmd
uv run python scripts/check_claude_evaluator.py --access cli
```

## 4. Run the final ten-question benchmark

The complete one-command run prepares missing frozen source pools, generates the 60 scripts, runs the independent Claude evaluation, exports CSV reports and creates PNG figures:

```cmd
uv run python scripts/run_final_study.py --claude-access cli
```

If the run is interrupted or incomplete, rerun the same command to resume the latest compatible incomplete Gemini experiment. Cases with hash-validated completed results are skipped; failed or unfinished cases are retried. If evaluation completed but analysis or figure generation failed, rerunning resumes reporting for that completed experiment without regenerating its cases. You can still select a specific run with `--experiment-id`.

To use the Anthropic API:

```cmd
uv run python scripts/run_final_study.py --claude-access api
```

Optional: build pools separately to inspect sources before generation:

```cmd
uv run python scripts/build_reference_pool.py --benchmark data/final_benchmark_10.jsonl
```

The builder writes `FQ01.json` to `FQ10.json` under the configured `REFERENCE_POOL_DIR` (default `data/reference_pools_final_10q`). Existing FQ01–FQ08 pools stay compatible; only FQ09 and FQ10 need building. You may copy *compatible* pools from an older version to this new directory; the builder validates question content and pool settings and rebuilds incompatible pools. Existing 10-question model generations are **not reusable** under the revised model or benchmark.

### Run the pilot ten-question dataset separately

Use a different experiment ID, and do not blend pilot scores with final study scores:

```cmd
uv run python scripts/run_final_study.py --benchmark data/pilot_benchmark_10.jsonl --claude-access cli
```

It uses Q01–Q08 and stores separate Q-prefixed frozen pools. The pilot is **not** the final-study test set.

### Smoke test before a full paid run

First prepare just the required frozen pool, then generate and evaluate two systems on one question:

```cmd
uv run python scripts/build_reference_pool.py --benchmark data/final_benchmark_10.jsonl --prompt-ids FQ01
uv run python scripts/run_benchmark.py --benchmark data/final_benchmark_10.jsonl --systems B2,P1 --prompt-ids FQ01 --require-existing-pools --claude-access cli
```

This is a pilot of the software, not one of the 60-case results. Use a fresh experiment ID for the main study.

### Resume an interrupted experiment

The command prints an experiment ID such as `EXP_GEMINI_...`. Pass exactly the same ID to resume after an interruption:

```cmd
uv run python scripts/run_final_study.py --experiment-id EXP_GEMINI_YOUR_ID --claude-access cli
```

Compatible completed cases are skipped only after integrity verification. You must not modify benchmark questions, model settings or frozen reference pools in the middle of a run.

## 5. Evaluation metrics

### Existing automatic metrics retained

- Query coverage (%): mean of continuous 0.0–1.0 Claude scores over the six-aspect checklist derived once per question and shared by all systems (bands in section 0).
- Query relevance (%), ten-dimension narration quality (%).
- Claim counts: supported, partial, unsupported, contradicted; strict supported-claim rate and weighted reference-evidence support (%).
- Citation precision, recall and F1 for emitted claim citations (not applicable to no-citation B1).
- Source counts and source-type diversity, retrieval calls, generation LLM calls, token usage, estimated generation cost, length, generation latency and LLM-call latency percentiles.
- Separate Claude evaluation time and estimated evaluation cost where usage telemetry is available.

### Exactly five added metrics

| Metric | Calculation and applicability | Interpretation |
|---|---|---|
| **Grounded Requirement Coverage (%)** | For each required aspect: `script_coverage_score × reference_support_score`, averaged over all six aspects. Claude independently scores support from the same frozen reference excerpts for all systems. | Teaching points must be both explained **and** reference-supported. N/A if no usable reference excerpts exist. |
| **Context Precision@5 (0–1)** | Number of Claude-judged relevant passages among the top `min(5, retrieved passages)` divided by that actual passage count. | Quality of evidence that was actually provided to the generator. N/A for B1 or any system with no retrieved passages. This is a judged precision proxy, not manually labeled precision. |
| **Context Recall (%)** | Claude rates whether the generation-side retrieved context provides enough information for each of the six checklist aspects; `0/0.5/1` per aspect, averaged. | **Aspect-based sufficiency proxy**, not classical corpus-wide retrieval recall. N/A if no retrieved passages exist. |
| **Repair Effectiveness (percentage points)** | `0.5 × (final coverage + final narration) − 0.5 × (pre-revision coverage + pre-revision narration)`. An additional Claude quality judgment is performed only for an actual changed pre-repair draft. | Improvement in **educational coverage and narration**, not change in factual accuracy or citation quality. N/A for unrevised cases. |
| **Requirement Preservation Rate (0–1)** | Among aspects rated `>0` before revision, proportion with a final aspect score **at least as high** as before. | Detects content loss during repair; N/A if no changed draft or no previously addressed aspects. |

All new ratings are automated Claude judgments and should not be represented as externally verified correctness or evidence truth. Retrieval metrics are only comparable where a system retrieved evidence. The final evaluation's *reference pool* is common for all six systems, while *generation-side retrieved passages* are architecture-dependent. Two distinct evidence inputs are therefore labelled separately in saved details.

The new fields are written into each `evaluations/<SYSTEM>/<QUESTION>.json` record, and into `analysis/evaluation_records.csv`, `analysis/system_summary.csv` and `analysis/per_question_results.csv`. Detailed per-aspect ratings, passage relevance and revision deltas are under each evaluation JSON's `additional_metric_details`.

**Cost caution:** Claude evaluates scripts with additional requests for the new measures; track evaluation overhead separately from generation cost. Unavailable scores stay blank/null rather than being filled with zero.

## 6. Results layout and research figures

```text
outputs_final_10q/
  experiments/
    EXP_GEMINI_.../
      manifest.json
      raw/B1/...P1/*.json
      evaluations/B1/...P1/*.json
      analysis/*.csv
      figures/*.png
```

Re-analyse or regenerate figures without new generation or evaluation calls:

```cmd
uv run python scripts/analyze_results.py --experiment-id EXP_GEMINI_YOUR_ID
uv run python scripts/generate_research_figures.py --experiment-id EXP_GEMINI_YOUR_ID
```

The project ships with six separate architecture PNG images:

```text
research_assets/figures/Architecture_B1.png
research_assets/figures/Architecture_B2.png
research_assets/figures/Architecture_B3.png
research_assets/figures/Architecture_B4.png
research_assets/figures/Architecture_B5.png
research_assets/figures/Architecture_P1.png
```

These are regenerated inside the experiment `figures/` folder too. **There is no Figure 02.** The old `Figure_02_proposed_architecture.png` is removed when plotting a reused figure directory.

The methodology diagrams are Figures 01, 03 and 04. Existing measured results remain Figures 05–20. New optional measured-result figures are:

- Figure 21: explained versus grounded requirements.
- Figure 22: context precision@5 and aspect-based context recall.
- Figure 23: quality change after actual repairs.
- Figure 24: requirement preservation after repairs.

A measured-result figure is written only when the required values exist. Figures are PNG only (250 DPI); no PDF graphics are exported. Static architecture images illustrate implementations, not empirical success rates.

## 7. Streamlit and validation

```cmd
uv run streamlit run app.py
uv run pytest -q
```

The Streamlit demo can use live retrieval, so it may not be identical to a frozen-corpus benchmark. It shows the separate P1 architecture PNG.

## 8. Research design and limitations

1. Use an identical fixed benchmark, generation model, Claude model, temperature/thinking settings and source-pool versions for all systems in one experiment.
2. Compare systems using matched questions. Report raw per-question results, means/medians/standard deviations and separate latency/cost data; avoid claiming causality from B1–B5 comparisons.
3. All automated evidence checks depend on the supplied references. A high evidence score is **not** the same as an independently verified factual truth score.
4. Context recall here is aspect coverage by retrieved passages, not recall against an exhaustive set of relevant documents. Context precision is judged by Claude, not human relevance labels.
5. Repair Effectiveness only covers educational completeness/narration, and is observed only for cases where a changed draft was saved. It should not be interpreted as a verified reduction in hallucinations.
6. Ten final questions are a **small exploratory sample**. Larger holdout datasets, repeated model runs, controlled P1 component ablations and independent reference answers would be needed for stronger publication claims. Run-to-run consistency, unsupported-claim density and cost per successful script are **not** added in this revision, following the five-metric limit.
7. No live 60-case Gemini/Claude study was run during code revision. Do not treat offline test data as research findings.

The architecture and evaluation choices are informed by LASEV (Yan et al., 2026), Instructional Agents (Yao et al., 2026) and Corrective Retrieval Augmented Generation (Yan et al., 2024). These are methodological references, not numerical comparison baselines for this project's script-quality scores.
