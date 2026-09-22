"""Phase 2 deterministic workflow tests."""
from datetime import datetime, timezone
import json
from types import SimpleNamespace
import pytest

from src.diagnostic import render_diagnostic
from src.extraction import extract_from_evidence
from src.models import (
    ClaimImportance,
    ClaimProposal,
    DiagnosticRun,
    EvidenceAssertion,
    EvidenceItem,
    EvidenceReference,
    ProspectIdentity,
    RunMetadata,
    RunStage,
    SourceType,
)
from src.providers.search import TavilySearchProvider
from src.providers.llm import LLMMessage, OllamaProvider
from src.research import public_identity_hint
from src.verification import verify_claim


def item(url: str, source_type: SourceType, excerpt: str, *, accessible: bool = True) -> EvidenceItem:
    return EvidenceItem(url=url, title="Public source", source_type=source_type,
                        retrieved_at=datetime.now(timezone.utc), supporting_excerpt=excerpt,
                        accessible=accessible)


def proposal_for(evidence: EvidenceItem, *, importance: ClaimImportance = ClaimImportance.STANDARD,
                 stance: str = "supports") -> ClaimProposal:
    return ClaimProposal(text="A public factual claim", importance=importance,
                         assertions=[EvidenceAssertion(evidence_id=evidence.evidence_id,
                                                       excerpt=evidence.supporting_excerpt, stance=stance)])


def identity_for(evidence: EvidenceItem) -> ProspectIdentity:
    reference = EvidenceReference(evidence_id=evidence.evidence_id, excerpt=evidence.supporting_excerpt)
    return ProspectIdentity(name="Jane Doe", company="Example Co", name_evidence=reference, company_evidence=reference)


def test_standard_claim_is_verified_from_exact_primary_excerpt() -> None:
    source = item("https://company.ae/about", SourceType.COMPANY_WEBSITE, "Jane Doe is the CEO.")
    claim, refused = verify_claim(proposal_for(source), [source])
    assert claim.status.value == "verified"
    assert claim.include_in_diagnostic is True
    assert refused is None


def test_important_claim_needs_two_independent_primary_domains() -> None:
    first = item("https://company.ae/news", SourceType.COMPANY_PRESS_RELEASE, "Fund closed its round.")
    second = item("https://regulator.gov.ae/register", SourceType.GOVERNMENT_OR_REGULATOR, "Fund closed its round.")
    proposal = ClaimProposal(text="Fund closed its round", importance=ClaimImportance.IMPORTANT,
        assertions=[EvidenceAssertion(evidence_id=first.evidence_id, excerpt=first.supporting_excerpt, stance="supports"),
                    EvidenceAssertion(evidence_id=second.evidence_id, excerpt=second.supporting_excerpt, stance="supports")])
    claim, _ = verify_claim(proposal, [first, second])
    assert claim.status.value == "verified"


def test_two_primary_records_from_one_domain_do_not_pass_second_source_check() -> None:
    first = item("https://company.ae/news", SourceType.COMPANY_PRESS_RELEASE, "Fund closed its round.")
    second = item("https://company.ae/about", SourceType.COMPANY_WEBSITE, "Fund closed its round.")
    proposal = ClaimProposal(text="Fund closed its round", importance=ClaimImportance.IMPORTANT,
        assertions=[EvidenceAssertion(evidence_id=first.evidence_id, excerpt=first.supporting_excerpt, stance="supports"),
                    EvidenceAssertion(evidence_id=second.evidence_id, excerpt=second.supporting_excerpt, stance="supports")])
    claim, _ = verify_claim(proposal, [first, second])
    assert claim.status.value == "partially_verified"


def test_secondary_evidence_cannot_independently_verify_important_claim() -> None:
    source = item("https://media.example/story", SourceType.SECONDARY_MEDIA, "Fund closed its round.")
    claim, _ = verify_claim(proposal_for(source, importance=ClaimImportance.IMPORTANT), [source])
    assert claim.status.value == "partially_verified"


def test_contradiction_refuses_claim() -> None:
    support = item("https://company.ae/about", SourceType.COMPANY_WEBSITE, "Jane is CEO.")
    conflict = item("https://regulator.gov.ae/register", SourceType.GOVERNMENT_OR_REGULATOR, "Jane is not CEO.")
    proposal = ClaimProposal(text="Jane is CEO", assertions=[
        EvidenceAssertion(evidence_id=support.evidence_id, excerpt=support.supporting_excerpt, stance="supports"),
        EvidenceAssertion(evidence_id=conflict.evidence_id, excerpt=conflict.supporting_excerpt, stance="contradicts"),
    ])
    claim, refused = verify_claim(proposal, [support, conflict])
    assert claim.status.value == "unverified"
    assert refused is not None and "Conflicting" in refused.reason


def test_inaccessible_and_nonmatching_quotes_do_not_verify() -> None:
    source = item("https://company.ae/about", SourceType.COMPANY_WEBSITE, "Jane is CEO.", accessible=False)
    claim, refused = verify_claim(proposal_for(source), [source])
    assert claim.status.value == "unverified"
    assert refused is not None


@pytest.mark.parametrize(
    ("url", "expected_hint"),
    [
        ("https://www.linkedin.com/in/example", "example"),
        ("https://linkedin.com/in/example", "example"),
        ("https://ae.linkedin.com/in/example", "example"),
        ("https://in.linkedin.com/in/example", "example"),
    ],
)
def test_linkedin_url_accepts_public_profile_hosts(url: str, expected_hint: str) -> None:
    assert public_identity_hint(url) == expected_hint


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/in/example",
        "https://www.linkedin.com/company/example",
    ],
)
def test_linkedin_url_requires_linkedin_profile_shape(url: str) -> None:
    with pytest.raises(ValueError):
        public_identity_hint(url)


def test_diagnostic_requires_human_approval() -> None:
    source = item("https://example.co/about", SourceType.COMPANY_WEBSITE, "Jane Doe leads Example Co.")
    run = DiagnosticRun(metadata=RunMetadata(linkedin_url="https://www.linkedin.com/in/jane", stage=RunStage.HUMAN_REVIEW), evidence=[source])
    with pytest.raises(ValueError):
        render_diagnostic(run, identity_for(source))


def test_approved_diagnostic_includes_an_illustrative_refusal_when_none_occurred() -> None:
    source = item("https://example.co/about", SourceType.COMPANY_WEBSITE, "Jane Doe leads Example Co.")
    run = DiagnosticRun(metadata=RunMetadata(linkedin_url="https://www.linkedin.com/in/jane",
                                              stage=RunStage.DIAGNOSTIC, human_review_approved=True), evidence=[source])
    assert "Example refused (illustrative" in render_diagnostic(run, identity_for(source))


def test_diagnostic_rejects_identity_not_grounded_in_an_exact_stored_excerpt() -> None:
    source = item("https://example.co/about", SourceType.COMPANY_WEBSITE, "Jane Doe leads Example Co.")
    identity = identity_for(source).model_copy(update={"name_evidence": EvidenceReference(evidence_id=source.evidence_id, excerpt="Jane is CEO")})
    run = DiagnosticRun(metadata=RunMetadata(linkedin_url="https://www.linkedin.com/in/jane",
                                              stage=RunStage.DIAGNOSTIC, human_review_approved=True), evidence=[source])
    with pytest.raises(ValueError, match="Identity fields require exact"):
        render_diagnostic(run, identity)


def test_tavily_failure_is_wrapped_for_visible_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingClient:
        def __init__(self, api_key: str) -> None:
            pass

        def search(self, **_: object) -> dict:
            raise OSError("network unavailable")

    import sys
    monkeypatch.setitem(sys.modules, "tavily", SimpleNamespace(TavilyClient=FailingClient))
    with pytest.raises(RuntimeError, match="Tavily public-web search failed"):
        TavilySearchProvider("test-key").search("Jane Doe UAE")


def test_malformed_llm_json_is_rejected_before_any_claim_is_returned() -> None:
    class MalformedProvider:
        def generate(self, *_: object, **__: object) -> SimpleNamespace:
            return SimpleNamespace(text="not valid json")

    with pytest.raises(ValueError):
        extract_from_evidence(MalformedProvider(), [])


def test_fenced_llm_json_is_extracted_before_deterministic_verification() -> None:
    class FencedProvider:
        def generate(self, *_: object, **__: object) -> SimpleNamespace:
            return SimpleNamespace(text="Here is the extraction:\n```json\n{\"identity\": null, \"claims\": []}\n```")

    identity, claims = extract_from_evidence(FencedProvider(), [])
    assert identity is None
    assert claims == []


def test_ollama_provider_uses_native_http_api_without_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def read(self) -> bytes:
            return b'{"response": "{}"}'

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_: object) -> None:
            return None

    def fake_urlopen(request: object, *, timeout: int) -> FakeResponse:
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        assert timeout == 300
        return FakeResponse()

    monkeypatch.setattr("src.providers.llm.urlopen", fake_urlopen)
    response = OllamaProvider("http://localhost:11434/v1", "qwen2.5:0.5b").generate([LLMMessage(role="user", content="hello")])
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["payload"] == {"model": "qwen2.5:0.5b", "prompt": "USER: hello", "stream": False,
                                   "options": {"temperature": 0}}
    assert response.text == "{}"



