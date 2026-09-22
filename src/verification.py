"""Deterministic evidence checks. No LLM code belongs in this module."""
from collections.abc import Iterable
from urllib.parse import urlparse

from .models import (
    Claim,
    ClaimImportance,
    ClaimProposal,
    ClaimStatus,
    EvidenceItem,
    RefusedClaim,
)


def _domain(item: EvidenceItem) -> str:
    return (urlparse(str(item.url)).hostname or "").lower().removeprefix("www.")


def _valid_assertions(proposal: ClaimProposal, evidence_by_id: dict) -> tuple[list[EvidenceItem], list[EvidenceItem]]:
    """Accept only accessible evidence whose stored excerpt contains the quote."""
    supporting: list[EvidenceItem] = []
    contradicting: list[EvidenceItem] = []
    for assertion in proposal.assertions:
        item = evidence_by_id.get(assertion.evidence_id)
        if not item or not item.accessible or assertion.excerpt not in item.supporting_excerpt:
            continue
        (supporting if assertion.stance == "supports" else contradicting).append(item)
    return supporting, contradicting


def verify_claim(proposal: ClaimProposal, evidence: Iterable[EvidenceItem]) -> tuple[Claim, RefusedClaim | None]:
    """Apply traceability, contradiction, source-quality, and independence rules."""
    evidence_by_id = {item.evidence_id: item for item in evidence}
    supporting, contradicting = _valid_assertions(proposal, evidence_by_id)
    evidence_ids = list(dict.fromkeys(item.evidence_id for item in [*supporting, *contradicting]))

    if contradicting:
        reason = "Conflicting accessible source evidence was found; the claim is unverified."
        return Claim(text=proposal.text, importance=proposal.importance, evidence_ids=evidence_ids,
                     verification_reason=reason), RefusedClaim(claim_text=proposal.text, reason=reason, evidence_ids=evidence_ids)

    primary = [item for item in supporting if item.is_eligible_primary]
    primary_domains = {_domain(item) for item in primary}
    if proposal.importance == ClaimImportance.IMPORTANT:
        if len(primary_domains) >= 2:
            status, reason = ClaimStatus.VERIFIED, "Supported by two independent eligible primary sources."
        elif supporting:
            status, reason = ClaimStatus.PARTIALLY_VERIFIED, "Important claim lacks two independent eligible primary sources."
        else:
            status, reason = ClaimStatus.UNVERIFIED, "No accessible, traceable supporting evidence was found."
    elif primary:
        status, reason = ClaimStatus.VERIFIED, "Supported by an accessible eligible primary source."
    elif supporting:
        status, reason = ClaimStatus.PARTIALLY_VERIFIED, "Supported only by a secondary public source; primary corroboration is needed."
    else:
        status, reason = ClaimStatus.UNVERIFIED, "No accessible, traceable supporting evidence was found."

    claim = Claim(text=proposal.text, importance=proposal.importance, status=status,
                  evidence_ids=evidence_ids, verification_reason=reason,
                  include_in_diagnostic=status != ClaimStatus.UNVERIFIED)
    refused = None
    if status == ClaimStatus.UNVERIFIED:
        refused = RefusedClaim(claim_text=proposal.text, reason=reason, evidence_ids=evidence_ids)
    return claim, refused


def verify_claims(proposals: Iterable[ClaimProposal], evidence: Iterable[EvidenceItem]) -> tuple[list[Claim], list[RefusedClaim]]:
    items = list(evidence)
    verified = [verify_claim(proposal, items) for proposal in proposals]
    return [claim for claim, _ in verified], [refusal for _, refusal in verified if refusal]
