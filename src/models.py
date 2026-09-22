"""Typed, serializable contracts for evidence-first diagnostic runs."""
from __future__ import annotations
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid4
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator

class SourceType(StrEnum):
    COMPANY_WEBSITE = "company_website"
    COMPANY_PRESS_RELEASE = "company_press_release"
    PORTFOLIO_COMPANY_ANNOUNCEMENT = "portfolio_company_announcement"
    GOVERNMENT_OR_REGULATOR = "government_or_regulator"
    STOCK_EXCHANGE = "stock_exchange"
    EVENT_ORGANIZER = "event_organizer"
    PERSONAL_WEBSITE = "personal_website"
    PUBLIC_LINKEDIN_PREVIEW = "public_linkedin_preview"
    SECONDARY_MEDIA = "secondary_media"
    DIRECTORY = "directory"
    SEARCH_RESULT = "search_result"
    OTHER = "other"

class ClaimStatus(StrEnum):
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"

class ClaimImportance(StrEnum):
    STANDARD = "standard"
    IMPORTANT = "important"

class RunStage(StrEnum):
    INTAKE = "intake"
    RESEARCH = "research"
    CLAIM_EXTRACTION = "claim_extraction"
    VERIFICATION = "verification"
    HUMAN_REVIEW = "human_review"
    DIAGNOSTIC = "diagnostic"
    COMPLETE = "complete"
    FAILED = "failed"

PRIMARY_SOURCE_TYPES = frozenset({SourceType.COMPANY_WEBSITE, SourceType.COMPANY_PRESS_RELEASE, SourceType.PORTFOLIO_COMPANY_ANNOUNCEMENT, SourceType.GOVERNMENT_OR_REGULATOR, SourceType.STOCK_EXCHANGE, SourceType.EVENT_ORGANIZER, SourceType.PERSONAL_WEBSITE, SourceType.PUBLIC_LINKEDIN_PREVIEW})

class EvidenceItem(BaseModel):
    """Immutable provenance record for a public source excerpt."""
    model_config = ConfigDict(frozen=True)
    evidence_id: UUID = Field(default_factory=uuid4)
    url: AnyHttpUrl
    title: Annotated[str, Field(min_length=1, max_length=500)]
    source_type: SourceType
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    supporting_excerpt: Annotated[str, Field(min_length=1, max_length=4000)]
    accessible: bool = True
    notes: str | None = None
    @property
    def is_eligible_primary(self) -> bool:
        return self.accessible and self.source_type in PRIMARY_SOURCE_TYPES

class Claim(BaseModel):
    """Candidate fact; a later deterministic engine owns its verification decision."""
    claim_id: UUID = Field(default_factory=uuid4)
    text: Annotated[str, Field(min_length=1, max_length=1000)]
    importance: ClaimImportance = ClaimImportance.STANDARD
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    evidence_ids: list[UUID] = Field(default_factory=list)
    verification_reason: str = "Not yet assessed by the deterministic verification engine."
    include_in_diagnostic: bool = False
    @model_validator(mode="after")
    def protect_inclusion(self) -> "Claim":
        if self.include_in_diagnostic and self.status == ClaimStatus.UNVERIFIED:
            raise ValueError("Unverified claims cannot be included in the diagnostic.")
        return self


class EvidenceAssertion(BaseModel):
    """An exact public-source excerpt proposed as support or contradiction.

    An extractor may propose assertions, but the verifier accepts one only when
    its quote occurs verbatim in the stored evidence item.
    """
    evidence_id: UUID
    excerpt: Annotated[str, Field(min_length=1, max_length=4000)]
    stance: Literal["supports", "contradicts"]


class EvidenceReference(BaseModel):
    """An exact excerpt that grounds a displayed identity field."""
    evidence_id: UUID
    excerpt: Annotated[str, Field(min_length=1, max_length=4000)]


class ClaimProposal(BaseModel):
    """Unverified candidate produced from public evidence only."""
    text: Annotated[str, Field(min_length=1, max_length=1000)]
    importance: ClaimImportance = ClaimImportance.STANDARD
    assertions: list[EvidenceAssertion] = Field(default_factory=list)


class ProspectIdentity(BaseModel):
    """Identity inferred from public-web evidence, never a LinkedIn fetch."""
    name: Annotated[str, Field(min_length=1, max_length=200)]
    company: Annotated[str, Field(min_length=1, max_length=300)]
    name_evidence: EvidenceReference
    company_evidence: EvidenceReference

class RefusedClaim(BaseModel):
    claim_text: Annotated[str, Field(min_length=1)]
    reason: Annotated[str, Field(min_length=1)]
    evidence_ids: list[UUID] = Field(default_factory=list)

class RunMetadata(BaseModel):
    run_id: UUID = Field(default_factory=uuid4)
    linkedin_url: AnyHttpUrl
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stage: RunStage = RunStage.INTAKE
    human_review_approved: bool = False

    @model_validator(mode="after")
    def require_human_review_before_diagnostic(self) -> "RunMetadata":
        if self.stage in {RunStage.DIAGNOSTIC, RunStage.COMPLETE} and not self.human_review_approved:
            raise ValueError("Human review approval is required before diagnostic generation.")
        return self

class DiagnosticRun(BaseModel):
    metadata: RunMetadata
    evidence: list[EvidenceItem] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    refused_claims: list[RefusedClaim] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
