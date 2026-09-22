"""Replaceable public-web search provider interface."""
from abc import ABC, abstractmethod
from urllib.parse import urlparse
from pydantic import AnyHttpUrl, BaseModel, Field

class SearchResult(BaseModel):
    title: str = Field(min_length=1)
    url: AnyHttpUrl
    snippet: str = ""

class SearchProvider(ABC):
    """Public discovery only; providers must never authenticate to LinkedIn."""
    provider_name: str
    @abstractmethod
    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        raise NotImplementedError

class TavilySearchProvider(SearchProvider):
    """Tavily public-web discovery; LinkedIn is never requested or returned."""
    provider_name = "tavily"
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        if "linkedin.com" in query.lower():
            raise ValueError("LinkedIn must not be queried by this application.")
        from tavily import TavilyClient

        try:
            response = TavilyClient(api_key=self.api_key).search(
                query=query, max_results=max_results, search_depth="basic", include_raw_content=False
            )
        except Exception as error:
            raise RuntimeError(f"Tavily public-web search failed: {error}") from error
        results = []
        for result in response.get("results", []):
            url = result.get("url", "")
            if urlparse(url).hostname and "linkedin.com" not in urlparse(url).hostname.lower():
                results.append(SearchResult(title=result.get("title") or url, url=url, snippet=result.get("content") or ""))
        return results
