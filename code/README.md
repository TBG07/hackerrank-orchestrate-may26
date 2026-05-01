# Support Triage Agent

A terminal-based AI agent that triages support tickets across **HackerRank**, **Claude (Anthropic)**, and **Visa** — using only the local support corpus. **No API key required.**

---

## Architecture

```
code/
├── main.py        Entry point — reads input CSV, calls agent, writes output CSV
├── agent.py       Core triage logic — domain detection, escalation, retrieval, response
├── retriever.py   Corpus loader + TF-IDF index (pure Python, zero external deps)
└── README.md      This file
```

### How it works

1. **Corpus loading** (`retriever.py`)
   All 774 markdown files under `data/` are loaded, split into overlapping 1200-character chunks, and indexed with a TF-IDF matrix built in pure Python. No vector DB, no embeddings API.

2. **Domain detection** (`agent.py`)
   The `company` field and issue text are scanned with keyword rules to identify which of the three domains (HackerRank / Claude / Visa) the ticket belongs to. This narrows retrieval to the relevant sub-corpus.

3. **Escalation check** (`agent.py`)
   Before any retrieval, a rule-based check flags tickets that must be escalated:
   - High-risk signals: identity theft, security vulnerabilities, bug bounties, legal threats, prompt injection attempts
   - Impossible requests: "restore my access", "increase my score", "ban the seller", etc.
   - Sensitive financial topics: fraud, unauthorized transactions
   - Site-wide outages

4. **Request type & product area classification** (`agent.py`)
   Keyword scoring assigns the ticket to a product area (e.g. `screen`, `interview`, `billing_claude`, `fraud_prevention`) and a request type (`product_issue`, `feature_request`, `bug`, `invalid`).

5. **Retrieval** (`retriever.py`)
   Top-6 corpus chunks are retrieved by TF-IDF score, domain-filtered first with a global fallback.

6. **Response extraction** (`agent.py`)
   The best paragraph from the retrieved chunks is selected by query-term overlap and returned as the grounded response.

7. **Output** (`main.py`)
   Results are written to `support_tickets/output.csv`.

### Design decisions

| Decision | Rationale |
|---|---|
| Pure TF-IDF, no vector DB | Zero infrastructure, fully deterministic, runs in ~1 second |
| No LLM / API key | Works offline; avoids hallucination risk; fully reproducible |
| Rule-based escalation | Explicit, auditable logic; no model needed for safety decisions |
| Domain-filtered retrieval | Improves precision by restricting search to the relevant sub-corpus |
| Escalation before retrieval | Safety checks run first, before any corpus lookup |

---

## Setup

### Requirements

- Python 3.9+
- No external packages needed (pure stdlib)

### Install (optional, for future LLM integration)

```bash
pip install -r requirements.txt
```

---

## Running the agent

### Process the full ticket set

```bash
python code/main.py
```

Output is written to `support_tickets/output.csv`.

### Test on the sample tickets

```bash
python code/main.py --sample
```

### Additional options

```
--input  PATH    Custom input CSV path
--output PATH    Custom output CSV path (default: support_tickets/output.csv)
--sample         Use sample_support_tickets.csv
--verbose        Print each ticket's response and justification to stdout
```

---

## Output schema

| Column | Allowed values |
|---|---|
| `status` | `replied`, `escalated` |
| `product_area` | Short label for the support category |
| `response` | User-facing answer grounded in the corpus |
| `justification` | Concise explanation of the routing decision |
| `request_type` | `product_issue`, `feature_request`, `bug`, `invalid` |

---

## Escalation logic

The agent escalates when any of the following apply:

- **High-risk signals**: identity theft, security vulnerabilities, bug bounties, legal threats, prompt injection / jailbreak attempts, destructive commands
- **Impossible requests**: actions the agent cannot perform (restore access, change scores, ban merchants, pause subscriptions, reschedule assessments)
- **Sensitive financial topics**: fraud, unauthorized transactions, identity theft
- **Site-wide outages**: "site is down", "none of the pages are accessible"
- **Vague tickets with no domain**: too little information to answer safely
