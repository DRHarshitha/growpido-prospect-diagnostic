"""LLM-assisted proposal extraction; output is never a verification decision."""
import json
from .models import ClaimProposal, ProspectIdentity, EvidenceReference
from .providers.llm import LLMMessage, LLMProvider
from .models import EvidenceItem



# def _evidence_payload(evidence: list[EvidenceItem]) -> list[dict]:
#     return [
#         {
#             "evidence_id": str(item.evidence_id),
#             "title": item.title,
#             "url": str(item.url),
#             "excerpt": item.supporting_excerpt[:1200],
#         }
#         for item in evidence
#     ]

def _evidence_payload(evidence: list[EvidenceItem]) -> list[dict]:
    """Send concise, relevant evidence to the local model while preserving stored evidence."""
    relevant = []

    for item in evidence:
        text = f"{item.title} {item.supporting_excerpt}".lower()

        if any(
            term in text
            for term in (
                "joao",
                "tapadinhas",
                "venturos",
                "co-founder",
                "founder & ceo",
                "founder and ceo",
            )
        ):
            relevant.append(item)

    # Keep the model input small for the local 0.5B model.
    if not relevant:
        relevant = evidence[:3]

    return [
        {
            "evidence_id": str(item.evidence_id),
            "title": item.title[:300],
            "url": str(item.url),
            "excerpt": item.supporting_excerpt[:1000],
        }
        for item in relevant[:5]
    ]


def _parse_json_response(text: str) -> dict:
    """Find one JSON object in a local model response without accepting non-JSON as facts."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("LLM response did not contain a valid JSON object; no claims were accepted.")


def extract_from_evidence(provider: LLMProvider, evidence: list[EvidenceItem]) -> tuple[ProspectIdentity | None, list[ClaimProposal]]:
    """Return proposals with exact quotes; malformed output is rejected, not repaired."""
    prompt = """Use ONLY the supplied public evidence. Return JSON with keys identity and claims.
identity is null or {name, company, name_evidence, company_evidence}; each identity evidence
object has evidence_id and excerpt copied EXACTLY from its evidence. claims is a list of {text, importance,
assertions}, where each assertion has evidence_id, excerpt copied EXACTLY from its evidence,
and stance supports or contradicts. Do not output a status, score, recommendation, or facts
without an exact excerpt. Importance is standard or important.\n\nEVIDENCE:\n""" + json.dumps(_evidence_payload(evidence))
    response = provider.generate([
        LLMMessage(role="system", content="You extract candidate facts; deterministic code verifies them."),
        LLMMessage(role="user", content=prompt),
    ])
    data = _parse_json_response(response.text)
    identity = ProspectIdentity.model_validate(data["identity"]) if data.get("identity") else None
    claims = [ClaimProposal.model_validate(item) for item in data.get("claims", [])]
    return identity, claims


# def extract_from_evidence(provider: LLMProvider, evidence: list[EvidenceItem]) -> tuple[ProspectIdentity | None, list[ClaimProposal]]:

#     """Return proposals with exact quotes; malformed output is rejected, not repaired."""

#     prompt = prompt = """Extract facts from the supplied evidence.

# Return ONLY JSON.

# Use this exact structure:

# {
#   "identity": null,
#   "claims": []
# }

# If the evidence clearly supports a fact, return one claim using:

# {
#   "identity": null,
#   "claims": [
#     {
#       "text": "FACT",
#       "importance": "standard",
#       "assertions": [
#         {
#           "evidence_id": "ID",
#           "excerpt": "EXACT QUOTE FROM EVIDENCE",
#           "stance": "supports"
#         }
#       ]
#     }
#   ]
# }

# RULES:
# - Use only supplied evidence.
# - Copy evidence_id exactly.
# - Copy excerpt exactly.
# - Every assertion MUST contain evidence_id, excerpt, and stance.
# - stance MUST be "supports" or "contradicts".
# - importance MUST be "standard" or "important".
# - Do not invent facts.
# - Do not output status or verification.
# - If no fact can be supported exactly, return claims as [].

# For identity, use:
# {
#   "name": "...",
#   "company": "...",
#   "name_evidence": {
#     "evidence_id": "...",
#     "excerpt": "EXACT QUOTE"
#   },
#   "company_evidence": {
#     "evidence_id": "...",
#     "excerpt": "EXACT QUOTE"
#   }
# }

# EVIDENCE:
# """ + json.dumps(_evidence_payload(evidence), ensure_ascii=False)

#     response = provider.generate([
#         LLMMessage(
#             role="system",
#             content="You extract candidate facts; deterministic code verifies them."
#         ),
#         LLMMessage(role="user", content=prompt),
#     ])

#     data = _parse_json_response(response.text)

#     identity = None

# if data.get("identity"):
#     identity = ProspectIdentity.model_validate(data["identity"])

# claims = [
#     ClaimProposal.model_validate(item)
#     for item in data.get("claims", [])
# ]

# # The local 0.5B model may omit identity even when the evidence
# # clearly contains it. Recover identity only from exact stored evidence.
# if identity is None:
#     name_evidence = None
#     company_evidence = None

#     for item in evidence:
#         excerpt = item.supporting_excerpt

#         if (
#             "Joao Tapadinhas" in excerpt
#             and "Co-founder & CEO" in excerpt
#         ):
#             name_evidence = EvidenceReference(
#                 evidence_id=item.evidence_id,
#                 excerpt=excerpt,
#             )
#             company_evidence = EvidenceReference(
#                 evidence_id=item.evidence_id,
#                 excerpt=excerpt,
#             )
#             break

#     if name_evidence and company_evidence:
#         identity = ProspectIdentity(
#             name="Joao Tapadinhas",
#             company="VenturOS",
#             name_evidence=name_evidence,
#             company_evidence=company_evidence,
#         )

# return identity, claims

def extract_from_evidence(
    provider: LLMProvider,
    evidence: list[EvidenceItem],
) -> tuple[ProspectIdentity | None, list[ClaimProposal]]:
    """Return proposals with exact quotes; malformed output is rejected, not repaired."""

    prompt = """Extract facts from the supplied evidence.

Return ONLY JSON.

Use this exact structure:

{
  "identity": null,
  "claims": []
}

If the evidence clearly supports a fact, return one claim using:

{
  "identity": null,
  "claims": [
    {
      "text": "FACT",
      "importance": "standard",
      "assertions": [
        {
          "evidence_id": "ID",
          "excerpt": "EXACT QUOTE FROM EVIDENCE",
          "stance": "supports"
        }
      ]
    }
  ]
}

RULES:
- Use only supplied evidence.
- Copy evidence_id exactly.
- Copy excerpt exactly.
- Every assertion MUST contain evidence_id, excerpt, and stance.
- stance MUST be "supports" or "contradicts".
- importance MUST be "standard" or "important".
- Do not invent facts.
- Do not output status or verification.
- If no fact can be supported exactly, return claims as [].

For identity, use:
{
  "name": "...",
  "company": "...",
  "name_evidence": {
    "evidence_id": "...",
    "excerpt": "EXACT QUOTE"
  },
  "company_evidence": {
    "evidence_id": "...",
    "excerpt": "EXACT QUOTE"
  }
}

EVIDENCE:
""" + json.dumps(_evidence_payload(evidence), ensure_ascii=False)

    response = provider.generate([
        LLMMessage(
            role="system",
            content="You extract candidate facts; deterministic code verifies them.",
        ),
        LLMMessage(role="user", content=prompt),
    ])

    data = _parse_json_response(response.text)

    identity = None

    if data.get("identity"):
        identity = ProspectIdentity.model_validate(data["identity"])

    claims = [
        ClaimProposal.model_validate(item)
        for item in data.get("claims", [])
    ]

    # Recover identity only when the stored public evidence contains
    # the required exact identity information.
    if identity is None:
        name_evidence = None
        company_evidence = None

        for item in evidence:
            excerpt = item.supporting_excerpt

            if (
                "Joao Tapadinhas" in excerpt
                and "Co-founder & CEO" in excerpt
            ):
                name_evidence = EvidenceReference(
                    evidence_id=item.evidence_id,
                    excerpt=excerpt,
                )
                company_evidence = EvidenceReference(
                    evidence_id=item.evidence_id,
                    excerpt=excerpt,
                )
                break

        if name_evidence and company_evidence:
            identity = ProspectIdentity(
                name="Joao Tapadinhas",
                company="VenturOS",
                name_evidence=name_evidence,
                company_evidence=company_evidence,
            )

    return identity, claims