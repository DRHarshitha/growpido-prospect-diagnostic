"""Phase 1 contract tests for verification and failure-handling scenarios."""
from datetime import datetime, timezone
import pytest
from pydantic import ValidationError
from src.models import Claim, ClaimImportance, ClaimStatus, EvidenceItem, RunMetadata, RunStage, SourceType
from src.providers.llm import LLMMessage, OpenAIProvider

def evidence(source_type: SourceType, *, accessible: bool = True) -> EvidenceItem:
    return EvidenceItem(url="https://example.com/source", title="Official source", source_type=source_type, retrieved_at=datetime.now(timezone.utc), supporting_excerpt="A directly observed public statement.", accessible=accessible)

def test_verified_claim_contract() -> None:
    claim = Claim(text="Person is CEO", status=ClaimStatus.VERIFIED, include_in_diagnostic=True)
    assert claim.status == ClaimStatus.VERIFIED

def test_partially_verified_claim_contract() -> None:
    claim = Claim(text="Person founded the company in 2019", status=ClaimStatus.PARTIALLY_VERIFIED)
    assert claim.status == ClaimStatus.PARTIALLY_VERIFIED

def test_unsupported_claim_is_excluded() -> None:
    claim = Claim(text="Person raised $20m", status=ClaimStatus.UNVERIFIED)
    assert claim.include_in_diagnostic is False
    with pytest.raises(ValidationError):
        Claim(text="Person raised $20m", status=ClaimStatus.UNVERIFIED, include_in_diagnostic=True)

def test_conflicting_sources_remain_unverified_contract() -> None:
    first, second = evidence(SourceType.COMPANY_WEBSITE), evidence(SourceType.COMPANY_PRESS_RELEASE)
    claim = Claim(text="Current job title", evidence_ids=[first.evidence_id, second.evidence_id])
    assert claim.status == ClaimStatus.UNVERIFIED

def test_inaccessible_source_is_not_eligible_primary() -> None:
    assert evidence(SourceType.COMPANY_WEBSITE, accessible=False).is_eligible_primary is False

def test_important_claim_with_one_source_starts_unverified() -> None:
    item = evidence(SourceType.COMPANY_WEBSITE)
    claim = Claim(text="Person manages the fund", importance=ClaimImportance.IMPORTANT, evidence_ids=[item.evidence_id])
    assert claim.status == ClaimStatus.UNVERIFIED

def test_llm_malformed_output_is_not_silently_accepted() -> None:
    with pytest.raises(ValidationError):
        LLMMessage.model_validate({"role": "assistant"})

def test_openai_provider_is_explicitly_deferred() -> None:
    with pytest.raises(NotImplementedError):
        OpenAIProvider().generate([LLMMessage(role="user", content="hello")])

def test_human_review_is_required_before_diagnostic_stage() -> None:
    with pytest.raises(ValidationError):
        RunMetadata(linkedin_url="https://www.linkedin.com/in/example", stage=RunStage.DIAGNOSTIC)
