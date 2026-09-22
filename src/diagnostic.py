"""Evidence-linked diagnostic rendering and scoped public-presence gap analysis."""
from dataclasses import dataclass

from .models import Claim, ClaimStatus, DiagnosticRun, EvidenceItem, EvidenceReference, ProspectIdentity, RunStage


@dataclass(frozen=True)
class PresenceGap:
    title: str
    finding: str
    evidence_ids: list[str]

def identify_presence_gaps(
    claims: list[Claim],
    evidence: list[EvidenceItem],
) -> list[PresenceGap]:
    """Identify scoped gaps from the reviewed evidence, not from claim wording alone."""

    usable_claims = [
        claim for claim in claims
        if claim.status != ClaimStatus.UNVERIFIED
    ]

    # Use both verified/partially-verified claim text and the actual
    # stored public evidence. This prevents generic LLM claim text
    # such as "FACT" from hiding relevant evidence.
    claim_corpus = " ".join(
        claim.text.lower()
        for claim in usable_claims
    )

    evidence_corpus = " ".join(
        item.supporting_excerpt.lower()
        for item in evidence
        if item.accessible
    )

    corpus = f"{claim_corpus} {evidence_corpus}"

    source_ids = [
        str(item.evidence_id)
        for item in evidence
        if item.accessible
    ]

    topics = [
        (
            "Leadership narrative",
            (
                "founder",
                "co-founder",
                "ceo",
                "chief executive",
                "managing partner",
                "partner",
                "director",
            ),
            "a clear leadership role",
        ),
        (
            "Company or fund proposition",
            (
                "company",
                "product",
                "service",
                "platform",
                "startup",
                "ai co-founder",
                "fund",
                "investment",
                "portfolio",
            ),
            "a clear organisation, product, service, or investment proposition",
        ),
        (
            "Independent third-party validation",
            (
                "award",
                "speaker",
                "conference",
                "press",
                "media",
                "regulator",
                "announcement",
                "customer",
                "client",
                "testimonial",
            ),
            "independent public validation",
        ),
    ]

    gaps: list[PresenceGap] = []

    for title, terms, label in topics:
        if not any(term in corpus for term in terms):
            gaps.append(
                PresenceGap(
                    title,
                    f"No verified or partially verified evidence of {label} appeared in the reviewed public-source set.",
                    source_ids,
                )
            )

    # If fewer than three genuine gaps are evidenced, use a neutral
    # evidence-depth gap rather than inventing a missing capability.
    while len(gaps) < 3:
        gaps.append(
            PresenceGap(
                "Evidence depth",
                "The reviewed public-source set does not yet provide enough traceable evidence to substantiate another public-presence theme.",
                source_ids,
            )
        )

    return gaps[:3]

# def identify_presence_gaps(claims: list[Claim], evidence: list[EvidenceItem]) -> list[PresenceGap]:
#     """Scoped gaps: absence in the reviewed public evidence, never a claim about reality."""
#     usable = [claim for claim in claims if claim.status != ClaimStatus.UNVERIFIED]
#     corpus = " ".join(claim.text.lower() for claim in usable)
#     source_ids = [str(item.evidence_id) for item in evidence]
#     topics = [
#         ("Leadership narrative", ("founder", "ceo", "managing", "partner", "director"), "a clear leadership role"),
#         ("Company or fund proposition", ("company", "fund", "invest", "product", "service", "portfolio"), "a clear organisation or investment proposition"),
#         ("Independent third-party validation", ("award", "speaker", "conference", "press", "regulator", "announcement"), "independent public validation"),
#     ]
#     gaps = []
#     for title, terms, label in topics:
#         if not any(term in corpus for term in terms):
#             gaps.append(PresenceGap(title, f"No verified or partially verified evidence of {label} appeared in the reviewed public-source set.", source_ids))
#     while len(gaps) < 3:
#         gaps.append(PresenceGap("Evidence depth", "The reviewed public-source set does not yet provide enough traceable evidence to substantiate another public-presence theme.", source_ids))
#     return gaps[:3]


def _identity_citation(assertion: EvidenceReference, evidence_by_id: dict) -> str:
    item = evidence_by_id.get(assertion.evidence_id)
    if not item or not item.accessible or assertion.excerpt not in item.supporting_excerpt:
        raise ValueError("Identity fields require exact, accessible stored evidence before diagnostic output.")
    return f"[{item.title}]({item.url})"


def render_diagnostic(run: DiagnosticRun, identity: ProspectIdentity) -> str:
    """Render only after a person explicitly approves the evidence review."""
    if not run.metadata.human_review_approved or run.metadata.stage not in {RunStage.DIAGNOSTIC, RunStage.COMPLETE}:
        raise ValueError("Human approval is required before final diagnostic output.")
    evidence_by_id = {item.evidence_id: item for item in run.evidence}
    name_source = _identity_citation(identity.name_evidence, evidence_by_id)
    company_source = _identity_citation(identity.company_evidence, evidence_by_id)
    lines = [f"# Prospect Diagnostic — {identity.name}", f"Name source: {name_source}",
             f"**Company / fund:** {identity.company} ({company_source})",
             "", "## Evidence-backed public profile"]
    included = [claim for claim in run.claims if claim.include_in_diagnostic]
    if included:
        for claim in included:
            citations = ", ".join(f"[{evidence_by_id[eid].title}]({evidence_by_id[eid].url})" for eid in claim.evidence_ids if eid in evidence_by_id)
            lines.append(f"- **{claim.status.value.replace('_', ' ').title()}** — {claim.text}  ")
            lines.append(f"  Evidence: {citations}. {claim.verification_reason}")
    else:
        lines.append("- No claim met the inclusion threshold in this reviewed evidence set.")
    lines.extend(["", "## Three public-presence gaps"])
    for gap in identify_presence_gaps(run.claims, run.evidence):
        lines.append(f"- **{gap.title}:** {gap.finding}")
    lines.extend(["", "## Refused claims"])
    if run.refused_claims:
        refusal = run.refused_claims[0]
        lines.append(f"- **Refused:** {refusal.claim_text} — {refusal.reason}")
    else:
        lines.append("- **Example refused (illustrative, not a prospect fact):** ‘The prospect raised a $20m round’ — refused because this run contains no exact accessible supporting excerpt.")
    lines.extend(["", "*Scope: public-web evidence reviewed for this run. The LinkedIn URL was used only as an identity reference and was never fetched.*"])
    return "\n".join(lines)
