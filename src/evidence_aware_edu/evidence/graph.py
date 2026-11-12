from __future__ import annotations

from dataclasses import dataclass, field

from evidence_aware_edu.schemas import Claim, ClaimVerification, Evidence


@dataclass
class ClaimEvidenceGraph:
    claims: dict[str, Claim] = field(default_factory=dict)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    edges: dict[str, set[str]] = field(default_factory=dict)

    def add_claims(self, claims: list[Claim]) -> None:
        for claim in claims:
            self.claims[claim.claim_id] = claim
            self.edges.setdefault(claim.claim_id, set())

    def add_evidence(self, evidence: list[Evidence]) -> None:
        for item in evidence:
            self.evidence[item.evidence_id] = item

    def apply_verifications(self, verifications: list[ClaimVerification]) -> None:
        for verification in verifications:
            self.edges.setdefault(verification.claim_id, set()).update(verification.evidence_ids)

    def evidence_for(self, claim_id: str) -> list[Evidence]:
        return [self.evidence[eid] for eid in self.edges.get(claim_id, set()) if eid in self.evidence]

    def to_dict(self) -> dict:
        return {
            "claims": [c.model_dump() for c in self.claims.values()],
            "evidence": [e.model_dump() for e in self.evidence.values()],
            "edges": {k: sorted(v) for k, v in self.edges.items()},
        }
