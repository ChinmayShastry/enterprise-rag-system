#!/usr/bin/env bash
# Shown on attach. This used to launch Streamlit immediately, which opened a
# preview of an app with an empty index — the first thing a new user saw was a
# warning. Printing the actual next steps is more useful.

cat <<'BANNER'

  ┌──────────────────────────────────────────────────────────────────┐
  │  Enterprise RAG System — container ready                         │
  └──────────────────────────────────────────────────────────────────┘

  1. Set your API key
       export OPENAI_API_KEY=sk-...
     (or add it once at github.com/settings/codespaces → Secrets)

  2. Add a PDF — drag one into the data/ folder in the file explorer

  3. Ingest it into a tenant
       python scripts/ingest.py --tenant acme --pdf "data/<your>.pdf" \
              --doc-id mydoc --title "My Document"

  4. Launch
       streamlit run app.py

  5. Verify the test suite (no API key needed)
       pytest

  Demo logins
       alice / alice123    admin   · full clearance + dashboard
       bob   / bob123      support · public + internal
       guest / guest123    viewer  · public only
       carol / carol123    admin   · DIFFERENT tenant (globex)

  guest and carol seeing nothing is correct — that is clearance and
  tenant isolation working, not a bug.

BANNER
