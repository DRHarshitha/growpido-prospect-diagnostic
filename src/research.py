"""Public-web research and evidence capture. LinkedIn is never fetched."""

from datetime import datetime, timezone
from urllib.parse import urlparse

from .models import EvidenceItem, SourceType
from .providers.search import SearchProvider


def public_identity_hint(linkedin_url: str) -> str:
    """Use only the URL slug as a search hint; do not request the URL itself."""
    parsed = urlparse(linkedin_url)
    host = (parsed.hostname or "").lower()

    if (
        (host != "linkedin.com" and not host.endswith(".linkedin.com"))
        or not parsed.path.startswith("/in/")
    ):
        raise ValueError("Provide a public LinkedIn profile URL in the /in/ form.")

    slug = parsed.path.removeprefix("/in/").strip("/")

    if not slug:
        raise ValueError("The LinkedIn URL must include a profile slug.")

    return slug.replace("-", " ")


def classify_source(url: str) -> SourceType:
    host = (urlparse(url).hostname or "").lower()

    if host.endswith("linkedin.com"):
        raise ValueError("LinkedIn sources are not accepted.")

    # Official VenturOS website
    if host == "ventur-os.com" or host.endswith(".ventur-os.com"):
        return SourceType.COMPANY_WEBSITE

    if any(token in host for token in ("gov.ae", ".gov", "dfsa", "adgm")):
        return SourceType.GOVERNMENT_OR_REGULATOR

    if any(
        token in host
        for token in ("eventbrite", "10times", "conference", "summit")
    ):
        return SourceType.EVENT_ORGANIZER

    if any(
        token in host
        for token in (
            "crunchbase",
            "wikipedia",
            "bloomberg",
            "forbes",
            "arabianbusiness",
        )
    ):
        return SourceType.SECONDARY_MEDIA

    return SourceType.OTHER


def search_public_evidence(
    search: SearchProvider,
    linkedin_url: str,
    *,
    name_hint: str = "",
    company_hint: str = "",
) -> list[EvidenceItem]:

    hint = " ".join(
        part
        for part in (
            name_hint.strip(),
            company_hint.strip(),
            public_identity_hint(linkedin_url),
        )
        if part
    )

    queries = [
        f'"{hint}" UAE',
        f'"{hint}" founder OR CEO OR fund manager',
    ]

    seen_urls: set[str] = set()
    evidence: list[EvidenceItem] = []

    for query in queries:
        for result in search.search(query, max_results=5):
            url = str(result.url)

            if (
                url in seen_urls
                or "linkedin.com" in (urlparse(url).hostname or "").lower()
                or not result.snippet.strip()
            ):
                continue

            seen_urls.add(url)

            evidence.append(
                EvidenceItem(
                    url=url,
                    title=str(result.title)[:500],
                    source_type=classify_source(url),
                    retrieved_at=datetime.now(timezone.utc),
                    supporting_excerpt=result.snippet,
                    notes="Public-web search result; verify excerpts before approving output.",
                )
            )

    return evidence