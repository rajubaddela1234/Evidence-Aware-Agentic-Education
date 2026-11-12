"""Six architecture figures, one for each implemented benchmark pipeline.

These diagrams are methodology illustrations, not measured experimental outcomes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# All diagrams share the same layout, so differences are easy to compare.
ARCHITECTURES = {
    "B1": {
        "name": "Direct Gemini (no retrieval)",
        "steps": ["Educational question", "Gemini direct generation", "Educational script"],
        "note": "One LLM call; no generation-side retrieval or revision",
    },
    "B2": {
        "name": "Single-source web RAG",
        "steps": ["Educational question", "Frozen web evidence", "Rank evidence excerpts", "Gemini synthesis", "Script + citations"],
        "note": "Single web source type; one generation call; no agentic repair",
    },
    "B3": {
        "name": "Fixed multi-source RAG",
        "steps": ["Educational question", "Frozen web + academic + video evidence", "Rank combined excerpts", "Gemini synthesis", "Script + citations"],
        "note": "Multiple source types; no iterative verification or rewrite",
    },
    "B4": {
        "name": "Original agentic baseline",
        "steps": ["Question / working query", "Gemini subquestion decomposition", "Frozen source retrieval", "Evidence ranking + synthesis", "Gemini feedback check", "Final educational script"],
        "note": "If feedback requests revision, rewrite query and repeat (bounded)",
        "loop": (4, 0),
    },
    "B5": {
        "name": "Agentic baseline with verification",
        "steps": ["B4 baseline pipeline", "Extract script claims", "Verify claims with evidence", "Whole-script repair", "Re-extract + verify if changed", "Final educational script"],
        "note": "B4 output followed by generation-side claim checks and repair",
    },
    "P1": {
        "name": "Requirement- and evidence-aware agentic system",
        "steps": ["Plan + complete requirements", "Adaptive retrieval / requirement audit", "Targeted requirement retrieval", "Claims + verify / targeted claim retrieval", "Contradiction checks + pedagogy plan", "Synthesize narration", "Final verification / requirements audit", "Repair if needed + final script"],
        "note": "Bounded corrective retrieval and final requirement-aware repair",
        "loop": (2, 1),
    },
}


def draw_architecture(system_id: str, target_dir: Path) -> Path:
    spec = ARCHITECTURES[system_id]
    steps = spec["steps"]
    fig, ax = plt.subplots(figsize=(13.0, max(5.0, len(steps) * 0.82 + 1.8)))
    ax.set_xlim(0, 12)
    ax.set_ylim(-1.0, len(steps) + 0.9)
    ax.axis("off")
    for idx, label in enumerate(steps):
        y = len(steps) - idx - .2
        rect = FancyBboxPatch((3.0, y - .36), 6, .66, boxstyle="round,pad=0.04",
                              linewidth=1.15, facecolor="#edf3f8", edgecolor="#39566e")
        ax.add_patch(rect)
        ax.text(6, y, label, ha="center", va="center", fontsize=10.6, color="#172e42")
        if idx:
            start_y = len(steps) - (idx - 1) - .55
            ax.add_patch(FancyArrowPatch((6, start_y), (6, y + .34), arrowstyle="-|>",
                                         mutation_scale=11, lw=1.35, color="#557086"))
    if "loop" in spec:
        end, start = spec["loop"]
        y_start = len(steps) - end - .2
        y_end = len(steps) - start - .2
        # Side route represents a conditional correction cycle, never an unconditional loop.
        ax.add_patch(FancyArrowPatch((9.1, y_start), (9.1, y_end), arrowstyle="-|>",
                                     connectionstyle="arc3,rad=-.33", mutation_scale=11,
                                     lw=1.5, linestyle="--", color="#8c6238"))
        ax.text(10.1, (y_start+y_end)/2, "optional\nrevision", fontsize=9,
                ha="center", va="center", color="#7a512e")
    ax.text(6, -.6, spec["note"], fontsize=9.5, ha="center", color="#4a6174")
    ax.set_title(f"{system_id} | {spec['name']}", fontsize=15, fontweight="medium", pad=15)
    target_dir.mkdir(parents=True, exist_ok=True)
    output = target_dir / f"Architecture_{system_id}.png"
    fig.savefig(output, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def generate_six_architectures(target_dir: Path) -> list[Path]:
    return [draw_architecture(system_id, target_dir) for system_id in ARCHITECTURES]


if __name__ == "__main__":
    output = Path("research_assets/figures")
    for path in generate_six_architectures(output):
        print(path)
