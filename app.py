"""Phase 2 public-web, evidence-first Prospect → Diagnostic workflow."""

import streamlit as st

from src.config import get_settings
from src.diagnostic import render_diagnostic
from src.extraction import extract_from_evidence
from src.models import DiagnosticRun, RunMetadata, RunStage, SourceType
from src.providers.llm import create_llm_provider
from src.providers.search import TavilySearchProvider
from src.research import search_public_evidence
from src.storage import LocalRunStore
from src.verification import verify_claims


st.set_page_config(
    page_title="Prospect to Diagnostic",
    page_icon="🔎",
    layout="wide",
)


def main() -> None:
    settings = get_settings()

    st.title("Prospect to Diagnostic")
    st.caption(
        "Phase 2 — public-web research, deterministic verification, "
        "and human approval"
    )

    st.info(
        "The LinkedIn URL is an identity reference only. "
        "This app never fetches LinkedIn, logs in, uses private data, "
        "bypasses controls, contacts people, or uses AWS."
    )

    # --------------------------------------------------------------
    # Active LLM
    # --------------------------------------------------------------

    provider_name = settings.llm_provider.lower()

    provider_label = (
        "Ollama"
        if provider_name == "ollama"
        else provider_name.title()
    )

    active_model = {
        "ollama": settings.ollama_model,
        "openai": settings.openai_model,
        "gemini": settings.gemini_model,
    }.get(provider_name, "unknown")

    st.caption(
        f"Active LLM: {provider_label} / {active_model}"
    )

    # --------------------------------------------------------------
    # Prospect input
    # --------------------------------------------------------------

    linkedin_url = st.text_input(
        "Public LinkedIn profile URL",
        placeholder="https://www.linkedin.com/in/example",
    )

    left, right = st.columns(2)

    name_hint = left.text_input(
        "Name hint (optional; improves public-web search)"
    )

    company_hint = right.text_input(
        "Company/fund hint (optional; improves public-web search)"
    )

    # ==============================================================
    # 1. PUBLIC-WEB RESEARCH
    # ==============================================================

    if st.button(
        "1. Research public sources",
        type="primary",
    ):
        # Local development uses .env.
        # Streamlit Cloud uses st.secrets.
        tavily_api_key = settings.tavily_api_key

        if not tavily_api_key:
            tavily_api_key = st.secrets.get(
                "TAVILY_API_KEY",
                "",
            )

        if not tavily_api_key:
            st.error(
                "Set TAVILY_API_KEY in local .env "
                "or Streamlit Cloud Secrets."
            )

        else:
            try:
                st.session_state.evidence = search_public_evidence(
                    TavilySearchProvider(tavily_api_key),
                    linkedin_url,
                    name_hint=name_hint,
                    company_hint=company_hint,
                )

                st.session_state.linkedin_url = linkedin_url
                st.session_state.identity = None
                st.session_state.claims = []
                st.session_state.refused_claims = []

                st.success(
                    f"Stored {len(st.session_state.evidence)} "
                    "public evidence records. "
                    "Review source types before extraction."
                )

            except (ValueError, RuntimeError) as error:
                st.error(str(error))

    # ==============================================================
    # 2. REVIEW STORED EVIDENCE
    # ==============================================================

    evidence = st.session_state.get(
        "evidence",
        [],
    )

    if evidence:
        st.subheader("2. Review stored evidence")

        revised = []

        for item in evidence:
            columns = st.columns([3, 2])

            columns[0].markdown(
                f"[{item.title}]({item.url})\n\n"
                f"{item.supporting_excerpt}"
            )

            source_types = list(SourceType)

            source_type = columns[1].selectbox(
                "Source type",
                source_types,
                index=source_types.index(
                    item.source_type
                ),
                key=str(item.evidence_id),
            )

            revised.append(
                item.model_copy(
                    update={
                        "source_type": source_type
                    }
                )
            )

        st.session_state.evidence = revised

        # ==========================================================
        # 3. EXTRACT IDENTITY + CANDIDATE CLAIMS
        # ==========================================================

        if st.button(
            "3. Extract identity and candidate claims from evidence"
        ):
            # ------------------------------------------------------
            # Check required LLM credentials
            # ------------------------------------------------------

            if (
                provider_name == "openai"
                and not settings.openai_api_key
            ):
                st.error(
                    "Set OPENAI_API_KEY in local .env "
                    "or Streamlit Secrets when "
                    "LLM_PROVIDER=openai. "
                    "No verification occurs in the LLM."
                )

            elif (
                provider_name == "gemini"
                and not (
                    settings.gemini_api_key
                    or st.secrets.get(
                        "GEMINI_API_KEY",
                        "",
                    )
                )
            ):
                st.error(
                    "Set GEMINI_API_KEY in local .env "
                    "or Streamlit Secrets when "
                    "LLM_PROVIDER=gemini."
                )

            else:
                try:
                    # --------------------------------------------------
                    # Gemini key:
                    # .env locally OR Streamlit Secrets in cloud
                    # --------------------------------------------------

                    gemini_api_key = settings.gemini_api_key

                    if not gemini_api_key:
                        gemini_api_key = st.secrets.get(
                            "GEMINI_API_KEY",
                            "",
                        )

                    provider = create_llm_provider(
                        provider=settings.llm_provider,
                        openai_api_key=settings.openai_api_key,
                        openai_model=settings.openai_model,
                        ollama_base_url=settings.ollama_base_url,
                        ollama_model=settings.ollama_model,
                        gemini_api_key=gemini_api_key,
                        gemini_model=settings.gemini_model,
                    )

                    # --------------------------------------------------
                    # LLM proposes identity/claims only.
                    # Verification remains deterministic.
                    # --------------------------------------------------

                    identity, proposals = extract_from_evidence(
                        provider,
                        revised,
                    )

                    claims, refusals = verify_claims(
                        proposals,
                        revised,
                    )

                    st.session_state.identity = identity
                    st.session_state.claims = claims
                    st.session_state.refused_claims = refusals

                    st.success(
                        f"Extracted {len(claims)} candidates; "
                        "verification used deterministic evidence rules."
                    )

                except Exception as error:
                    st.error(
                        "Extraction failed; no claims were accepted: "
                        f"{error}"
                    )

    # ==============================================================
    # 4. DETERMINISTIC VERIFICATION REVIEW
    # ==============================================================

    claims = st.session_state.get(
        "claims",
        [],
    )

    identity = st.session_state.get(
        "identity"
    )

    if claims:
        st.subheader(
            "4. Deterministic verification review"
        )

        for claim in claims:
            st.write(
                f"**{claim.status.value.replace('_', ' ').title()}** "
                f"— {claim.text}"
            )

            st.caption(
                claim.verification_reason
            )

        # ==========================================================
        # 5. REQUIRED HUMAN APPROVAL
        # ==========================================================

        st.subheader(
            "5. Required human approval"
        )

        approved = st.checkbox(
            "I reviewed the stored evidence, claim statuses, "
            "contradictions, and refused claims; approve final "
            "diagnostic generation."
        )

        if approved and st.button(
            "Generate approved one-page diagnostic"
        ):
            if not identity:
                st.error(
                    "No public-evidence identity was extracted; "
                    "do not generate a diagnostic."
                )

            else:
                run = DiagnosticRun(
                    metadata=RunMetadata(
                        linkedin_url=st.session_state.linkedin_url,
                        stage=RunStage.DIAGNOSTIC,
                        human_review_approved=True,
                    ),
                    evidence=st.session_state.evidence,
                    claims=claims,
                    refused_claims=st.session_state.get(
                        "refused_claims",
                        [],
                    ),
                )

                LocalRunStore(
                    settings.runs_directory
                ).save(run)

                report = render_diagnostic(
                    run,
                    identity,
                )

                st.subheader(
                    "Approved Prospect Diagnostic"
                )

                st.markdown(report)

                st.download_button(
                    "Export diagnostic (.md)",
                    report,
                    file_name="prospect-diagnostic.md",
                    mime="text/markdown",
                )


if __name__ == "__main__":
    main()