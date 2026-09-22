
# Prospect to Diagnostic

An evidence-first Streamlit prototype for Growpido's Track B take-home assignment.

## Track

**Track B — Prospect to Diagnostic**

The system takes a public LinkedIn URL only as an identity reference, researches publicly available web sources, captures traceable evidence, extracts candidate claims, verifies those claims using deterministic rules, identifies public-presence gaps, and generates a one-page prospect diagnostic after human approval.

---

## Demo Prospect

The prototype was tested using:

**Joao Tapadinhas — Co-founder & CEO, VenturOS**

Public LinkedIn reference:

https://ae.linkedin.com/in/jtapadinhas

Official company source:

https://ventur-os.com/about

The LinkedIn URL is used only as an identity reference.

The application does not log into LinkedIn, scrape LinkedIn, or access private LinkedIn information.

---

## Architecture

```text
Public LinkedIn URL
        |
        v
Public Web Research
        |
        v
Evidence Capture
        |
        v
Claim / Identity Extraction
        |
        v
Deterministic Verification
        |
        v
Second-Check Validation
        |
        v
Public-Presence Gap Analysis
        |
        v
Human Review
        |
        v
One-Page Diagnostic