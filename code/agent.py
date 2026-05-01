"""
agent.py — Core triage logic (no external API required).

For each ticket:
  1. Detect domain (hackerrank / claude / visa / unknown)
  2. Classify request type via keyword rules
  3. Assess escalation need via risk signals
  4. Retrieve top corpus chunks via TF-IDF
  5. Extract the best passage as the grounded response
  6. Return all 5 output fields
"""

from __future__ import annotations

import re
from typing import Optional

from retriever import retrieve

# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS = {
    "hackerrank": [
        "hackerrank", "hacker rank", "coding test", "assessment", "test invite",
        "mock interview", "screen", "interview platform", "candidate", "recruiter",
        "test score", "plagiarism", "proctoring", "skillup", "engage", "chakra",
        "library", "question bank", "hiring", "ats integration",
    ],
    "claude": [
        "claude", "anthropic", "claude.ai", "claude api", "claude pro", "claude team",
        "claude enterprise", "claude code", "claude desktop", "claude mobile",
        "artifacts", "projects", "skills", "cowork", "claude subscription",
        "bedrock", "claude model", "claude workspace",
    ],
    "visa": [
        "visa card", "visa payment", "visa transaction", "visa traveller",
        "visa cheque", "visa debit", "visa credit", "visa merchant",
        "visa dispute", "visa fraud", "visa stolen", "visa lost",
        "visa atm", "visa refund", "visa chargeback", "visa rules",
        "visa support", "visa india", "visa gcas",
    ],
}


def detect_domain(company: str, issue: str) -> Optional[str]:
    """Return 'hackerrank', 'claude', 'visa', or None."""
    c = (company or "").strip().lower()
    if "hackerrank" in c:
        return "hackerrank"
    if "claude" in c or "anthropic" in c:
        return "claude"
    if "visa" in c:
        return "visa"

    text = (issue or "").lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return domain
    return None


# ---------------------------------------------------------------------------
# Product area classification
# ---------------------------------------------------------------------------

_PRODUCT_AREAS = {
    # HackerRank
    "screen": [
        "test", "assessment", "invite", "candidate", "score", "plagiarism",
        "proctoring", "question", "coding challenge", "test variant", "retake",
        "time accommodation", "extra time", "test expir", "test active",
        "cut off", "cutoff", "test result", "test report",
    ],
    "interview": [
        "interview", "interviewer", "zoom", "video call", "audio", "screen share",
        "inactivity", "lobby", "virtual lobby", "interview room", "interview link",
    ],
    "integrations": [
        "greenhouse", "workday", "lever", "ashby", "ats", "integration",
        "sso", "saml", "okta", "azure", "scim", "jit provisioning",
        "slack", "zapier", "google calendar", "outlook calendar",
    ],
    "skillup": [
        "skillup", "learning", "course", "certification", "skill",
        "practice", "apply tab", "submission", "challenge",
    ],
    "engage": [
        "engage", "event", "hackathon", "leaderboard", "microsite",
        "broadcast email", "campaign",
    ],
    "chakra": [
        "chakra", "ai interviewer", "ai interview",
    ],
    "community": [
        "community", "account", "delete account", "password", "login",
        "sign in", "sign up", "google login", "github login", "profile",
        "certificate", "resume",
    ],
    "billing_hackerrank": [
        "subscription", "billing", "payment", "invoice", "plan", "refund",
        "order id", "cs_live",
    ],
    # Claude
    "account_management": [
        "account", "delete account", "email address", "password", "login",
        "session", "sign out", "log out", "export data", "coupon", "promotion",
    ],
    "billing_claude": [
        "billing", "subscription", "pro plan", "max plan", "team plan",
        "enterprise plan", "invoice", "payment", "refund", "usage limit",
        "usage bundle", "seat",
    ],
    "features": [
        "artifact", "project", "skill", "cowork", "web search", "research",
        "extended thinking", "memory", "file", "upload", "image", "voice",
        "dictation", "excel", "powerpoint", "word", "office",
    ],
    "privacy": [
        "privacy", "data", "conversation", "delete conversation", "incognito",
        "sensitive", "personal data", "training data", "crawl",
    ],
    "api_console": [
        "api", "api key", "console", "rate limit", "token", "context window",
        "prompt", "bedrock", "workbench", "workspace",
    ],
    "claude_code": [
        "claude code", "code review", "xcode", "foundry",
    ],
    "security_claude": [
        "security", "vulnerability", "bug bounty", "compromised", "stolen",
        "hack", "breach",
    ],
    "education": [
        "education", "university", "lti", "canvas", "professor", "student",
        "college",
    ],
    # Visa
    "lost_stolen_card": [
        "lost card", "stolen card", "lost or stolen", "card stolen",
        "card lost", "report lost", "block card", "emergency card",
    ],
    "fraud_prevention": [
        "fraud", "fraudulent", "scam", "unauthorized", "identity theft",
        "identity stolen", "suspicious transaction",
    ],
    "dispute_resolution": [
        "dispute", "chargeback", "wrong product", "merchant", "refund",
        "charge", "transaction issue",
    ],
    "travel_support": [
        "travel", "abroad", "overseas", "atm", "currency", "exchange rate",
        "gcas", "global customer", "emergency cash",
    ],
    "travelers_cheques": [
        "traveller cheque", "traveler cheque", "travellers cheque",
        "travelers cheque", "cheque serial",
    ],
    "visa_rules": [
        "visa rules", "minimum spend", "minimum amount", "maximum limit",
        "merchant rules", "visa policy",
    ],
    "general_support": [
        "visa card", "visa support", "card declined", "atm locator",
        "cash withdrawal", "urgent cash",
    ],
}


def classify_product_area(issue: str, subject: str, domain: Optional[str]) -> str:
    text = (f"{subject} {issue}").lower()

    # Score each area
    best_area, best_score = "general", 0
    for area, keywords in _PRODUCT_AREAS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_area = area

    # Domain-level fallback
    if best_score == 0:
        fallback = {
            "hackerrank": "screen",
            "claude": "account_management",
            "visa": "general_support",
        }
        best_area = fallback.get(domain or "", "general")

    return best_area


# ---------------------------------------------------------------------------
# Request type classification
# ---------------------------------------------------------------------------

_BUG_SIGNALS = [
    "not working", "doesn't work", "broken", "error", "crash", "down",
    "failing", "failed", "bug", "glitch", "issue", "problem", "blocker",
    "stopped working", "can't access", "cannot access", "unable to",
    "not loading", "not responding", "all requests failing",
]

_FEATURE_SIGNALS = [
    "feature request", "would like to", "can you add", "please add",
    "suggestion", "enhancement", "improve", "wish", "want to see",
    "could you", "is it possible to add",
]

_INVALID_SIGNALS = [
    "thank you", "thanks", "hello", "hi there", "good morning",
    "what is the name", "who is", "tell me about", "explain",
    "what year", "how many people",
]


def classify_request_type(issue: str, subject: str, domain: Optional[str]) -> str:
    text = (f"{subject} {issue}").lower()

    # Bug signals take priority — even for unknown domain (e.g. "site is down")
    if any(sig in text for sig in _BUG_SIGNALS):
        return "bug"
    if any(sig in text for sig in _FEATURE_SIGNALS):
        return "feature_request"

    # Out-of-scope / irrelevant — only after ruling out bugs/features
    if domain is None:
        if any(sig in text for sig in _INVALID_SIGNALS):
            return "invalid"
        if not any(
            kw in text
            for kws in _DOMAIN_KEYWORDS.values()
            for kw in kws
        ):
            return "invalid"

    return "product_issue"


# ---------------------------------------------------------------------------
# Escalation logic
# ---------------------------------------------------------------------------

# High-risk: always escalate
_ESCALATE_ALWAYS = [
    "identity theft", "identity stolen", "identity has been stolen",
    "account takeover", "data breach", "security vulnerability",
    "bug bounty", "major security",
    "legal action", "lawsuit", " sue ", "court", "police",
    "death", "harm", "threat", "abuse",
    "hack my", "hacked my",
    # Prompt injection / jailbreak attempts
    "ignore previous", "ignore all", "system prompt", "internal rules",
    "show me your prompt", "reveal your instructions", "affiche toutes",
    "logique exacte", "documents récupérés", "règles internes",
    "delete all files", "rm -rf", "format c:",
]

# Requests the agent cannot fulfil — escalate
_ESCALATE_CANNOT_DO = [
    "restore my access", "restore access",
    "increase my score", "change my score", "review my answers",
    "tell the company to", "move me to the next round",
    "ban the seller", "ban the merchant",
    "make visa refund", "force a refund",
    "fill in the forms", "infosec process",
    "pause our subscription", "cancel our subscription",
    "update my certificate", "update my name on the certificate",
    "reschedule my", "rescheduling",
    "give me the refund", "refund asap",
    "give me my money",
]

# Financial / sensitive — escalate unless corpus clearly covers it
# NOTE: lost/stolen card, traveller's cheques, dispute, urgent cash are
# all covered by the Visa corpus — do NOT auto-escalate those.
_ESCALATE_SENSITIVE = [
    "fraud", "fraudulent transaction", "unauthorized transaction",
    "identity theft", "identity stolen",
]

# Out-of-scope topics — reply as invalid, don't escalate
_OUT_OF_SCOPE = [
    "iron man", "actor", "movie", "film", "celebrity",
    "weather", "stock price", "sports score",
    "recipe", "cook", "restaurant",
    "delete all files", "rm -rf",
]


def should_escalate(issue: str, subject: str, domain: Optional[str]) -> tuple[bool, str]:
    """
    Returns (escalate: bool, reason: str).
    """
    text = (f"{subject} {issue}").lower()

    # Prompt injection / jailbreak / destructive commands — escalate immediately
    for sig in _ESCALATE_ALWAYS:
        if sig in text:
            return True, f"High-risk signal detected: '{sig}'"

    # Requests the agent cannot fulfil
    for sig in _ESCALATE_CANNOT_DO:
        if sig in text:
            return True, f"Request requires human action: '{sig}'"

    # Sensitive financial/security topics
    for sig in _ESCALATE_SENSITIVE:
        if sig in text:
            return True, f"Sensitive topic requiring human review: '{sig}'"

    # Site-wide outage — always escalate regardless of domain
    if any(s in text for s in ["site is down", "none of the pages are accessible", "nothing is working"]):
        return True, "Possible site-wide outage — escalate for investigation"

    # Completely vague tickets with no domain — can't answer safely
    if domain is None and len(text.strip()) < 30:
        return True, "Ticket is too vague and has no identifiable product domain — escalating for human review"

    return False, ""


# ---------------------------------------------------------------------------
# Response extraction from corpus
# ---------------------------------------------------------------------------

def _clean_markdown(text: str) -> str:
    """Strip markdown syntax for a cleaner user-facing response."""
    # Remove YAML front matter
    text = re.sub(r"^---.*?---\s*", "", text, flags=re.DOTALL)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove image links
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Convert links to plain text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove heading hashes but keep text
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_best_passage(chunks: list[dict], query: str, max_chars: int = 1200) -> str:
    """
    From the top retrieved chunks, extract the most relevant passage.
    Prefers paragraphs that contain query terms.
    """
    if not chunks:
        return ""

    query_tokens = set(re.sub(r"[^a-z0-9\s]", " ", query.lower()).split())

    best_passage = ""
    best_score = -1

    for chunk in chunks[:4]:
        cleaned = _clean_markdown(chunk["text"])
        # Split into paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", cleaned) if len(p.strip()) > 60]
        for para in paragraphs:
            para_tokens = set(re.sub(r"[^a-z0-9\s]", " ", para.lower()).split())
            overlap = len(query_tokens & para_tokens)
            if overlap > best_score:
                best_score = overlap
                best_passage = para

    if not best_passage and chunks:
        best_passage = _clean_markdown(chunks[0]["text"])

    # Trim to max_chars at a sentence boundary
    if len(best_passage) > max_chars:
        trimmed = best_passage[:max_chars]
        last_period = trimmed.rfind(".")
        if last_period > max_chars // 2:
            trimmed = trimmed[: last_period + 1]
        best_passage = trimmed

    return best_passage


# ---------------------------------------------------------------------------
# Canned responses for special cases
# ---------------------------------------------------------------------------

_OUT_OF_SCOPE_REPLY = (
    "I'm sorry, this request is outside the scope of what I can help with. "
    "I can only assist with support queries related to HackerRank, Claude, and Visa. "
    "Please contact the appropriate support channel for your query."
)

_ESCALATION_REPLY = (
    "Thank you for reaching out. Your request requires attention from a human support agent "
    "and has been escalated accordingly. A member of our team will follow up with you shortly."
)

_GENERIC_REPLY = (
    "Thank you for contacting support. Based on the information available, "
    "I was unable to find a specific answer in our knowledge base. "
    "Please contact our support team directly for further assistance."
)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_ticket(issue: str, subject: str, company: str) -> dict:
    """
    Process one support ticket and return all 5 output fields.
    Works entirely offline — no API key required.
    """
    issue = (issue or "").strip()
    subject = (subject or "").strip()
    company_raw = (company or "").strip()
    company = "" if company_raw.lower() in ("none", "") else company_raw

    query = f"{subject} {issue}".strip()

    # 1. Detect domain
    domain = detect_domain(company, issue)

    # 2. Check escalation FIRST (before invalid check — site-down etc. must escalate)
    escalate, escalation_reason = should_escalate(issue, subject, domain)
    if escalate:
        request_type = classify_request_type(issue, subject, domain)
        product_area = classify_product_area(issue, subject, domain)
        return {
            "status": "escalated",
            "product_area": product_area,
            "response": _ESCALATION_REPLY,
            "justification": escalation_reason,
            "request_type": request_type,
        }

    # 3. Classify request type
    request_type = classify_request_type(issue, subject, domain)

    # 4. Classify product area
    product_area = classify_product_area(issue, subject, domain)

    # 5. Check for out-of-scope (invalid with no domain)
    if request_type == "invalid":
        return {
            "status": "replied",
            "product_area": product_area,
            "response": _OUT_OF_SCOPE_REPLY,
            "justification": (
                "The ticket does not relate to HackerRank, Claude, or Visa support topics. "
                "Replied with an out-of-scope message."
            ),
            "request_type": "invalid",
        }

    # 6. Retrieve corpus chunks
    chunks = retrieve(query, top_k=6, domain_filter=domain)
    if len(chunks) < 2:
        # Fallback: global search
        chunks = retrieve(query, top_k=6, domain_filter=None)

    # 7. Extract grounded response
    if chunks:
        response = _extract_best_passage(chunks, query)
        source_title = chunks[0]["title"]
        justification = (
            f"Answered using corpus content from '{source_title}' "
            f"(domain: {chunks[0]['domain']}). "
            f"Top retrieval score: {chunks[0]['score']:.3f}."
        )
    else:
        response = _GENERIC_REPLY
        justification = "No relevant corpus content found; provided generic guidance."

    if not response:
        response = _GENERIC_REPLY

    return {
        "status": "replied",
        "product_area": product_area,
        "response": response,
        "justification": justification,
        "request_type": request_type,
    }
