from __future__ import annotations

from evidence_aware_edu.agents.common import AgentBase
from evidence_aware_edu.schemas import Claim, ClaimVerification, Evidence, VerificationBundle
from evidence_aware_edu.utils import clip

COMPLETION_RETRIES = 2  # extra calls for claims the verifier omitted (evaluator only)
VERIFY_BATCH_SIZE = 10  # claims per verifier call
VERIFY_MAX_OUTPUT_TOKENS = 6000


class IncompleteVerification(RuntimeError):
    """The verifier kept omitting claims; the case is an evaluation error, not a low score."""


class VerifierAgent(AgentBase):
    SYSTEM = """You are a strict claim-evidence verifier.
Judge each claim only against the supplied evidence. Do not use outside knowledge.
Use 'supported' only when the evidence directly entails the important content of the claim.
Use 'partial' when only part is supported or wording is stronger than evidence.
Use 'contradicted' when evidence directly conflicts. Otherwise use 'unsupported'.
For supported/partial/contradicted claims, return ONLY exact evidence IDs that appear in the supplied
EVIDENCE block and that actually justify that decision. Do not invent evidence IDs.
For unsupported claims return an empty evidence_ids list.
Return exactly ONE verification for EVERY claim. Never duplicate a claim ID and never omit one.
Confidence is confidence in your evidence-support judgement, NOT confidence that the claim is
factually true in the world. Evidence support and factual accuracy are different concepts.
Return structured output only."""

    ALLOWED_EVIDENCE_NOTE = (
        "Each claim lists its ALLOWED EVIDENCE. Judge each claim ONLY against its own allowed "
        "passages and ignore every other passage for that claim, even if another passage would "
        "support it. A claim whose allowed passages do not support it is 'unsupported', and "
        "evidence_ids may only contain that claim's allowed IDs.\n\n"
    )

    @staticmethod
    def _prompt(
        claims: list[Claim],
        evidence: list[Evidence],
        max_chars: int,
        allowed_by_claim: dict[str, set[str]] | None = None,
    ) -> str:
        if allowed_by_claim is None:
            claim_text = "\n".join(f"{c.claim_id}: {c.text}" for c in claims)
            note = ""
        else:
            claim_text = "\n".join(
                f"{c.claim_id}: {c.text}\n   ALLOWED EVIDENCE: "
                + (", ".join(sorted(allowed_by_claim.get(c.claim_id, set()))) or "NONE")
                for c in claims
            )
            note = VerifierAgent.ALLOWED_EVIDENCE_NOTE
        parts: list[str] = []
        used = 0
        for e in evidence:
            item = f"[{e.evidence_id}] {e.title}\n{clip(e.passage, 1800)}\n"
            if used + len(item) > max_chars:
                break
            parts.append(item)
            used += len(item)
        return f"{note}CLAIMS:\n{claim_text}\n\nEVIDENCE:\n" + "\n".join(parts)

    @staticmethod
    def _normalize(
        claims: list[Claim],
        evidence: list[Evidence],
        returned: list[ClaimVerification],
        allowed_by_claim: dict[str, set[str]] | None = None,
    ) -> list[ClaimVerification]:
        """Clean verifier output and keep one valid decision per claim.

        With ``allowed_by_claim`` a claim may only be justified by its own allowed evidence IDs;
        a supported/partial decision that cites none of them is downgraded to unsupported.
        """
        claim_ids = {c.claim_id for c in claims}
        supplied_ids = {e.evidence_id for e in evidence}
        grouped: dict[str, list[ClaimVerification]] = {}
        for item in returned:
            if item.claim_id in claim_ids:
                grouped.setdefault(item.claim_id, []).append(item)

        # Prefer the more cautious label when duplicate decisions disagree.
        severity = {
            "supported": 0,
            "partial": 1,
            "unsupported": 2,
            "contradicted": 3,
        }
        normalized: list[ClaimVerification] = []
        for claim in claims:
            options = grouped.get(claim.claim_id, [])
            if not options:
                normalized.append(
                    ClaimVerification(
                        claim_id=claim.claim_id,
                        status="unsupported",
                        confidence=0.0,
                        evidence_ids=[],
                        rationale="No verification was returned; treated conservatively as unsupported.",
                    )
                )
                continue

            chosen = max(options, key=lambda x: (severity[x.status], -x.confidence))
            valid_evidence_ids = (
                supplied_ids
                if allowed_by_claim is None
                else supplied_ids & set(allowed_by_claim.get(claim.claim_id, set()))
            )
            valid_ids = []
            seen = set()
            for eid in chosen.evidence_ids:
                if eid in valid_evidence_ids and eid not in seen:
                    valid_ids.append(eid)
                    seen.add(eid)

            if chosen.status == "unsupported":
                valid_ids = []
            elif not valid_ids:
                chosen = ClaimVerification(
                    claim_id=claim.claim_id,
                    status="unsupported",
                    confidence=0.0,
                    evidence_ids=[],
                    rationale=(
                        "Verifier returned no usable supplied evidence ID for a non-unsupported "
                        "decision; downgraded conservatively to unsupported."
                    ),
                )
                normalized.append(chosen)
                continue

            chosen.claim_id = claim.claim_id
            chosen.evidence_ids = valid_ids
            normalized.append(chosen)

        return normalized

    def run(
        self,
        claims: list[Claim],
        evidence: list[Evidence],
        task: str = "verify_claims",
        max_chars: int | None = None,
        *,
        allowed_by_claim: dict[str, set[str]] | None = None,
        require_complete: bool = False,
    ) -> list[ClaimVerification]:
        """Verify claims against evidence.

        ``allowed_by_claim`` restricts each claim to its own evidence IDs (citation checking).
        ``require_complete`` (used by the evaluator) re-asks for claims the model omitted and
        raises :class:`IncompleteVerification` if any are still missing, instead of silently
        scoring them as unsupported. Generation-side callers keep the conservative default.
        """
        if not claims:
            return []
        limit = max_chars or self.llm.settings.max_context_chars
        returned: list[ClaimVerification] = []
        attempts = 1 + (COMPLETION_RETRIES if require_complete else 0)
        # Verify in batches: one JSON answer for 20-30 claims with rationales can exceed the
        # output-token limit and arrive cut off mid-string, which fails to parse.
        for start in range(0, len(claims), VERIFY_BATCH_SIZE):
            batch = claims[start : start + VERIFY_BATCH_SIZE]
            pending = list(batch)
            for _ in range(attempts):
                result, log = self.llm.generate_structured(
                    task=task,
                    system_instruction=self.SYSTEM,
                    prompt=self._prompt(pending, evidence, max_chars=limit,
                                        allowed_by_claim=allowed_by_claim),
                    schema=VerificationBundle,
                    max_output_tokens=VERIFY_MAX_OUTPUT_TOKENS,
                )
                self._record(log)
                returned.extend(result.verifications)
                answered = {v.claim_id for v in returned}
                pending = [c for c in batch if c.claim_id not in answered]
                if not pending:
                    break
            if require_complete and pending:
                raise IncompleteVerification(
                    f"{task}: verifier omitted {len(pending)} claim(s) after {attempts} attempts: "
                    + ", ".join(c.claim_id for c in pending)
                )
        return self._normalize(claims, evidence, returned, allowed_by_claim)
