import json
import html as html_lib
import os
import re
import time
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from difflib import SequenceMatcher

import requests
import pandas as pd
import streamlit as st
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Customer Support Lab",
    page_icon="🤖",
    layout="wide",
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

MODEL_NAME = "gpt-4.1-mini"
MAX_QUERY_LENGTH = 2000

# Local UI assets — rendered without any API calls
APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
EXPERIMENT_ARCHITECTURES = {
    "fixed": ASSETS_DIR / "experiment1_architecture.png",
    "dynamic": ASSETS_DIR / "experiment2_architecture.png",
}

openai_client = (
    OpenAI(api_key=OPENAI_API_KEY)
    if OPENAI_API_KEY
    else None
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def validate_configuration():
    missing = []

    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")

    if not SERPER_API_KEY:
        missing.append("SERPER_API_KEY")

    return missing


def parse_json_response(raw_text: str):
    cleaned = raw_text.strip()

    cleaned = re.sub(
        r"^```json\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"^```\s*",
        "",
        cleaned,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    return json.loads(cleaned.strip())


def clamp_score(value):
    try:
        value = int(value)
    except Exception:
        value = 0

    return max(0, min(100, value))


def clear_run_state():
    keys = [
        "security",
        "non_agentic",
        "agentic",
        "evidence",
        "evaluation",
        "observations",
        "run_mode",
        "run_metrics",
        "experiment",
        "dynamic_non_agentic",
        "dynamic_agentic",
        "dynamic_observations",
        "shared_workflow_search",
        "experiment1_validation",
        "dynamic_action_validation",
        "dynamic_decision_validation",
        "dynamic_evaluation",
        "latest_evaluation",
        "experiment_notice",
        "experiment_block_notice",
    ]

    for key in keys:
        st.session_state.pop(key, None)



# ============================================================
# TEMPORAL CONTEXT + DETERMINISTIC TEMPORAL VALIDATION
# NO OPENAI / SEARCH CALLS
# ============================================================

def current_date_context():
    now = datetime.now()
    return {
        "iso": now.strftime("%Y-%m-%d"),
        "readable": now.strftime("%B %d, %Y"),
        "year": now.year,
    }


# ============================================================
# SESSION-SCOPED API REUSE
# Reuses safe evidence/classification results within this browser session.
# No external cache/database required.
# ============================================================

def normalize_query_for_cache(query: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]", " ", (query or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _cache_scope() -> str:
    # Date-scope all external evidence so time-sensitive answers are never
    # carried across days. Session state already limits reuse to this session.
    return current_date_context()["iso"]


def _cache_lookup(cache_name: str, query: str, allow_near: bool = False, threshold: float = 0.96):
    target = normalize_query_for_cache(query)
    if not target:
        return None, None

    scope = _cache_scope()
    entries = st.session_state.get(cache_name, [])

    # Exact reuse first.
    for entry in reversed(entries):
        if entry.get("scope") == scope and entry.get("normalized") == target:
            return deepcopy(entry.get("value")), "exact"

    # Conservative near-duplicate reuse is only used for SEARCH EVIDENCE.
    # It is intentionally not used for safety classification or model answers.
    if allow_near:
        best = None
        best_ratio = 0.0
        for entry in entries:
            if entry.get("scope") != scope:
                continue
            ratio = SequenceMatcher(None, target, entry.get("normalized", "")).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best = entry
        if best is not None and best_ratio >= threshold:
            return deepcopy(best.get("value")), f"near:{best_ratio:.2f}"

    return None, None


def _cache_store(cache_name: str, query: str, value):
    normalized = normalize_query_for_cache(query)
    if not normalized:
        return

    scope = _cache_scope()
    entries = st.session_state.setdefault(cache_name, [])

    # Replace exact same entry instead of growing the session cache endlessly.
    entries[:] = [
        entry for entry in entries
        if not (entry.get("scope") == scope and entry.get("normalized") == normalized)
    ]
    entries.append({
        "scope": scope,
        "normalized": normalized,
        "query": query,
        "value": deepcopy(value),
    })

    # Small bounded session cache is enough for this lab.
    if len(entries) > 30:
        del entries[:-30]



# ============================================================
# LIVE PROCESSING TRACKER
# Local UI only — zero OpenAI/search calls.
# ============================================================

BUILDATHON_PROGRESS_STAGES = [
    ("guardrails", "Guardrails"),
    ("assistant", "Assistant"),
    ("search", "Web Search"),
    ("entry", "Entry Agent"),
    ("output_guardrail", "Output Guardrail"),
    ("evaluation", "Evaluation"),
]


def _buildathon_progress_html(phase: str = "idle", message: str = ""):
    """
    Render a compact animated workflow tracker.

    IMPORTANT:
    CrewAI exposes the three-agent execution to this Streamlit page as one
    blocking kickoff. Therefore Assistant, Web Search and Entry are animated
    together during the CrewAI phase instead of pretending we can observe an
    internal sub-stage that CrewAI has not exposed.

    HTML is intentionally emitted as one compact line so Streamlit's Markdown
    parser cannot interpret indented nested <div> tags as a code block.
    """
    phase = phase or "idle"

    stage_states = {
        key: "pending"
        for key, _ in BUILDATHON_PROGRESS_STAGES
    }

    badge_text = "READY"
    badge_class = "idle"

    if phase == "guardrails":
        stage_states["guardrails"] = "active"
        badge_text = "RUNNING"
        badge_class = "running"

    elif phase == "crew":
        stage_states["guardrails"] = "done"
        stage_states["assistant"] = "active"
        stage_states["search"] = "active"
        stage_states["entry"] = "active"
        badge_text = "RUNNING"
        badge_class = "running"

    elif phase == "output_guardrail":
        for key in ["guardrails", "assistant", "search", "entry"]:
            stage_states[key] = "done"
        stage_states["output_guardrail"] = "active"
        badge_text = "RUNNING"
        badge_class = "running"

    elif phase == "evaluation":
        for key in [
            "guardrails",
            "assistant",
            "search",
            "entry",
            "output_guardrail",
        ]:
            stage_states[key] = "done"
        stage_states["evaluation"] = "active"
        badge_text = "RUNNING"
        badge_class = "running"

    elif phase == "complete":
        for key, _ in BUILDATHON_PROGRESS_STAGES:
            stage_states[key] = "done"
        badge_text = "COMPLETE"
        badge_class = "complete"

    elif phase == "cached":
        for key, _ in BUILDATHON_PROGRESS_STAGES:
            stage_states[key] = "done"
        badge_text = "CACHE REUSED"
        badge_class = "cached"

    elif phase == "blocked":
        stage_states["guardrails"] = "blocked"
        badge_text = "BLOCKED"
        badge_class = "blocked"

    elif phase == "output_blocked":
        for key in ["guardrails", "assistant", "search", "entry"]:
            stage_states[key] = "done"
        stage_states["output_guardrail"] = "blocked"
        badge_text = "OUTPUT BLOCKED"
        badge_class = "blocked"

    pieces = []

    for index, (key, label) in enumerate(BUILDATHON_PROGRESS_STAGES):
        state = stage_states[key]

        pieces.append(
            f'<div class="live-stage {state}">'
            f'<div class="live-stage-dot">'
            f'<span class="live-stage-check">✓</span>'
            f'</div>'
            f'<div class="live-stage-label">{html_lib.escape(label)}</div>'
            f'</div>'
        )

        if index < len(BUILDATHON_PROGRESS_STAGES) - 1:
            next_key = BUILDATHON_PROGRESS_STAGES[index + 1][0]
            next_state = stage_states[next_key]

            if state == "done" and next_state in {"done", "active"}:
                line_state = "done"
            elif state == "active" or next_state == "active":
                line_state = "active"
            else:
                line_state = "pending"

            pieces.append(
                f'<div class="live-stage-line {line_state}"><span></span></div>'
            )

    safe_message = html_lib.escape(
        message
        or (
            "Ready to run the guarded three-agent workflow."
            if phase == "idle"
            else ""
        )
    )

    return (
        '<div class="live-progress-shell">'
        '<div class="live-progress-head">'
        '<div>'
        '<span class="live-progress-kicker">LIVE PROCESSING</span>'
        '<span class="live-progress-title">Workflow stage</span>'
        '</div>'
        f'<span class="live-progress-badge {badge_class}">{badge_text}</span>'
        '</div>'
        '<div class="live-progress-track">'
        + "".join(pieces)
        + '</div>'
        f'<div class="live-progress-message">{safe_message}</div>'
        '</div>'
    )


def render_buildathon_progress(slot, phase: str = "idle", message: str = ""):
    slot.markdown(
        _buildathon_progress_html(phase, message),
        unsafe_allow_html=True,
    )


def _latest_cached_value(cache_name: str):
    """Return the latest value for today's session-scoped cache."""
    scope = _cache_scope()
    entries = st.session_state.get(cache_name, [])

    for entry in reversed(entries):
        if entry.get("scope") == scope:
            return deepcopy(entry.get("value"))

    return None



# ============================================================
# BUILDATHON SESSION MEMORY
# Local Streamlit session state only — no separate API/model call.
# Experiments intentionally remain memory-isolated so comparisons stay controlled.
# ============================================================

SESSION_MEMORY_MAX_TURNS = 4
SESSION_MEMORY_MAX_ANSWER_CHARS = 700


def _memory_store():
    return st.session_state.setdefault(
        "conversation_memory",
        {"buildathon": []},
    )


def get_session_memory(scope: str = "buildathon"):
    return list(_memory_store().get(scope, []))


def clear_session_memory(scope: str = "buildathon"):
    _memory_store()[scope] = []


def add_session_memory_turn(
    user_query: str,
    assistant_answer: str,
    scope: str = "buildathon",
):
    """Store one completed conversational turn in browser-session memory."""
    query = (user_query or "").strip()
    answer = (assistant_answer or "").strip()

    if not query or not answer:
        return

    memory = _memory_store().setdefault(scope, [])

    # Do not grow memory when an identical completed turn is repeated.
    if memory:
        last = memory[-1]
        if (
            normalize_query_for_cache(last.get("user", ""))
            == normalize_query_for_cache(query)
            and last.get("assistant", "").strip() == answer
        ):
            return

    memory.append({
        "user": query,
        "assistant": answer[:SESSION_MEMORY_MAX_ANSWER_CHARS],
    })

    # Rolling bounded memory keeps prompt/token usage controlled.
    if len(memory) > SESSION_MEMORY_MAX_TURNS:
        del memory[:-SESSION_MEMORY_MAX_TURNS]


def build_session_memory_context(
    scope: str = "buildathon",
    max_turns: int = SESSION_MEMORY_MAX_TURNS,
):
    """
    Build compact conversational context for model prompts.

    No summarization/model call is used. The most recent turns are inserted
    directly and are intentionally bounded to control token usage.
    """
    turns = get_session_memory(scope)[-max_turns:]

    if not turns:
        return ""

    blocks = []
    for index, turn in enumerate(turns, start=1):
        blocks.append(
            f"Previous turn {index}:\n"
            f"Customer: {turn.get('user', '')}\n"
            f"Assistant: {turn.get('assistant', '')}"
        )

    return "\n\n".join(blocks)


def build_search_memory_context(scope: str = "buildathon", max_turns: int = 2):
    """
    Compact search-only context for follow-up resolution.

    Includes the prior customer request plus a short slice of the final answer,
    so references such as "it", "that version" or "when was it released" can
    resolve to the previously discussed entity. No model call is used.
    """
    turns = get_session_memory(scope)[-max_turns:]
    blocks = []

    for turn in turns:
        user_text = turn.get("user", "").strip()
        answer_text = turn.get("assistant", "").strip()

        if not user_text:
            continue

        if answer_text:
            answer_text = answer_text[:320]
            blocks.append(
                f"Previous customer request: {user_text}. "
                f"Previous final answer: {answer_text}"
            )
        else:
            blocks.append(f"Previous customer request: {user_text}")

    return " | ".join(blocks)


def extract_explicit_dates(text: str):
    """Extract common explicit calendar dates in textual order."""
    if not text:
        return []

    month_names = (
        "January|February|March|April|May|June|July|August|"
        "September|October|November|December"
    )

    patterns = [
        rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*|\s+)(\d{{4}})\b",
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_names})\s+(\d{{4}})\b",
        r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b",
    ]

    matches = []

    for pattern_index, pattern in enumerate(patterns):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            try:
                if pattern_index == 0:
                    month, day, year = match.groups()
                    parsed = datetime.strptime(
                        f"{month} {day} {year}",
                        "%B %d %Y",
                    ).date()
                elif pattern_index == 1:
                    day, month, year = match.groups()
                    parsed = datetime.strptime(
                        f"{day} {month} {year}",
                        "%d %B %Y",
                    ).date()
                else:
                    year, month, day = match.groups()
                    parsed = datetime(
                        int(year),
                        int(month),
                        int(day),
                    ).date()

                matches.append({
                    "date": parsed,
                    "text": match.group(0),
                    "position": match.start(),
                })
            except Exception:
                pass

    matches.sort(key=lambda item: item["position"])
    return matches


def run_temporal_validation(
    query: str,
    final_answer: str,
    selected_event_date: str | None = None,
):
    """
    Deterministic temporal validation for relative-date requests.

    Preferred path:
    - The Entry Agent returns a machine-readable selected event date as part of
      its existing output. The application strips that metadata before display.
      This adds zero extra LLM/search calls.

    Fallback path:
    - If no selected event date is available, inspect explicit dates in the
      final answer and choose the first valid future date.
    """
    q = (query or "").lower()

    temporal_terms = (
        "next",
        "upcoming",
        "nearest",
        "when is",
        "when's",
        "coming",
    )

    applied = any(term in q for term in temporal_terms)

    if not applied:
        return {
            "applied": False,
            "passed": True,
            "status": "NOT_REQUIRED",
            "detail": "No relative-date selection check required.",
            "current_date": current_date_context()["iso"],
            "selected_date": None,
            "selection_source": "NOT_REQUIRED",
            "dates_found": [],
        }

    current_date = datetime.now().date()
    dates = extract_explicit_dates(final_answer)

    if selected_event_date:
        try:
            selected = datetime.strptime(
                selected_event_date.strip(),
                "%Y-%m-%d",
            ).date()

            passed = (
                selected >= current_date
                if "today" in q
                else selected > current_date
            )

            return {
                "applied": True,
                "passed": passed,
                "status": "PASS" if passed else "FAIL",
                "detail": (
                    f"Selected event date {selected.isoformat()} is after "
                    f"current date {current_date.isoformat()}."
                    if passed and selected > current_date
                    else
                    f"Selected event date {selected.isoformat()} is valid for "
                    f"current date {current_date.isoformat()}."
                    if passed
                    else
                    f"Selected event date {selected.isoformat()} is not a valid "
                    f"future date relative to {current_date.isoformat()}."
                ),
                "current_date": current_date.isoformat(),
                "selected_date": selected.isoformat(),
                "selection_source": "ENTRY_AGENT_SELECTED_EVENT_DATE",
                "dates_found": [
                    item["date"].isoformat()
                    for item in dates
                ],
            }
        except ValueError:
            pass

    if not dates:
        return {
            "applied": True,
            "passed": False,
            "status": "FAIL",
            "detail": (
                "Relative-date request detected, but neither a valid selected "
                "event date nor an explicit calendar date was available."
            ),
            "current_date": current_date.isoformat(),
            "selected_date": None,
            "selection_source": "FALLBACK_TEXT_EXTRACTION",
            "dates_found": [],
        }

    asks_today = "today" in q
    valid_candidates = [
        item for item in dates
        if (
            item["date"] >= current_date
            if asks_today
            else item["date"] > current_date
        )
    ]

    if valid_candidates:
        selected = valid_candidates[0]
        return {
            "applied": True,
            "passed": True,
            "status": "PASS",
            "detail": (
                f"Fallback temporal check selected {selected['date'].isoformat()} "
                f"relative to current date {current_date.isoformat()}."
            ),
            "current_date": current_date.isoformat(),
            "selected_date": selected["date"].isoformat(),
            "selection_source": "FALLBACK_TEXT_EXTRACTION",
            "dates_found": [
                item["date"].isoformat()
                for item in dates
            ],
        }

    latest_found = max(item["date"] for item in dates)
    return {
        "applied": True,
        "passed": False,
        "status": "FAIL",
        "detail": (
            f"No valid future event date was found. The latest explicit date "
            f"is {latest_found.isoformat()}, while current date is "
            f"{current_date.isoformat()}."
        ),
        "current_date": current_date.isoformat(),
        "selected_date": None,
        "selection_source": "FALLBACK_TEXT_EXTRACTION",
        "dates_found": [
            item["date"].isoformat()
            for item in dates
        ],
    }


# ============================================================
# LAYER 1 — DETERMINISTIC GUARDRAILS
# NO OPENAI CALL
# ============================================================

def run_deterministic_guardrails(query: str):
    query = query.strip()
    checks = []

    query_present = bool(query)

    checks.append(
        {
            "name": "Query validation",
            "status": (
                "PASS"
                if query_present
                else "BLOCK"
            ),
            "detail": (
                "Query provided."
                if query_present
                else "Query is empty."
            ),
        }
    )

    length_valid = (
        query_present
        and len(query) <= MAX_QUERY_LENGTH
    )

    checks.append(
        {
            "name": "Input length",
            "status": (
                "PASS"
                if length_valid
                else "BLOCK"
            ),
            "detail": (
                f"{len(query)} / "
                f"{MAX_QUERY_LENGTH} characters."
            ),
        }
    )

    injection_patterns = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?prior\s+instructions",
        r"disregard\s+(all\s+)?previous\s+instructions",
        r"override\s+(your\s+)?instructions",
        r"forget\s+(all\s+)?previous\s+instructions",
        r"reveal\s+(your\s+)?system\s+prompt",
        r"show\s+(me\s+)?(your\s+)?system\s+prompt",
        r"print\s+(your\s+)?system\s+prompt",
        r"reveal\s+(your\s+)?hidden\s+instructions",
        r"show\s+(your\s+)?hidden\s+instructions",
    ]

    injection_detected = any(
        re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        )
        for pattern in injection_patterns
    )

    checks.append(
        {
            "name": "Prompt injection",
            "status": (
                "BLOCK"
                if injection_detected
                else "PASS"
            ),
            "detail": (
                "Potential instruction-manipulation "
                "pattern detected."
                if injection_detected
                else
                "No known injection pattern detected."
            ),
        }
    )

    secret_patterns = [
        (
            r"(show|reveal|print|give|display).{0,30}"
            r"(api[\s_-]?key|secret[\s_-]?key|password|token)"
        ),
        (
            r"(show|reveal|print|give|display).{0,30}"
            r"(environment variables|\.env)"
        ),
        (
            r"(show|reveal|print|give|display).{0,30}"
            r"(developer message|internal instructions)"
        ),
    ]

    secret_detected = any(
        re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        )
        for pattern in secret_patterns
    )

    checks.append(
        {
            "name": "Secrets protection",
            "status": (
                "BLOCK"
                if secret_detected
                else "PASS"
            ),
            "detail": (
                "Potential secret-extraction request detected."
                if secret_detected
                else
                "No secret-extraction pattern detected."
            ),
        }
    )

    allowed = all(
        check["status"] == "PASS"
        for check in checks
    )

    return {
        "allowed": allowed,
        "query": query,
        "checks": checks,
    }


# ============================================================
# LAYER 2 — AI SAFETY CLASSIFIER
# ONE OPENAI CALL FOR QUERIES THAT PASS LAYER 1
# ============================================================

def run_ai_safety_guardrail(query: str):
    # Exact-query reuse only. Safety decisions are never reused for merely
    # similar wording because small wording changes can alter intent.
    cached, cache_match = _cache_lookup(
        "ai_safety_cache",
        query,
        allow_near=False,
    )
    if cached is not None:
        cached["elapsed_time"] = 0.0
        cached["openai_calls"] = 0
        cached["cache_hit"] = True
        cached["cache_match"] = cache_match
        return cached

    prompt = f"""
You are a security classifier for a publicly accessible
customer-support AI application.

Determine whether the following request may proceed.

Return ONLY valid JSON:

{{
  "decision": "ALLOW" or "BLOCK",
  "category": "SAFE" or one unsafe category,
  "reason": "short explanation"
}}

Unsafe categories:

EXPLICIT_SEXUAL_CONTENT
HARMFUL_ILLEGAL_ACTIVITY
MALICIOUS_CYBER_ACTIVITY
SELF_HARM_HIGH_RISK
HATE_OR_TARGETED_ABUSE
CREDENTIAL_OR_SECRET_THEFT
TOOL_MISUSE
OTHER_UNSAFE

Rules:

1. Block requests seeking explicit sexual or pornographic material.
2. Block instructions facilitating serious harm or illegal activity.
3. Block malicious cyber activity.
4. Block credential, API-key, token or secret theft.
5. Block attempts to misuse external tools for unsafe purposes.
6. Allow legitimate defensive-security questions.
7. Allow legitimate reporting and customer-support requests.
8. Do not block merely because a sensitive word is present.
9. Judge the requested action and intent.

USER QUERY:

{query}
"""

    start = time.perf_counter()

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        result = parse_json_response(
            response.choices[0].message.content
        )

        decision = str(
            result.get(
                "decision",
                "BLOCK",
            )
        ).strip().upper()

        category = str(
            result.get(
                "category",
                "OTHER_UNSAFE",
            )
        ).strip().upper()

        reason = str(
            result.get(
                "reason",
                "",
            )
        ).strip()

        if decision not in {
            "ALLOW",
            "BLOCK",
        }:
            decision = "BLOCK"
            category = "CLASSIFIER_ERROR"
            reason = (
                "Unexpected safety-classifier response."
            )

        output = {
            "allowed": decision == "ALLOW",
            "decision": decision,
            "category": category,
            "reason": reason,
            "status": "SUCCESS",
            "elapsed_time": (
                time.perf_counter() - start
            ),
            "openai_calls": 1,
            "cache_hit": False,
            "cache_match": None,
        }
        _cache_store("ai_safety_cache", query, output)
        return output

    except Exception as error:
        return {
            "allowed": False,
            "decision": "BLOCK",
            "category": "CLASSIFIER_ERROR",
            "reason": (
                "Safety classification failed. "
                "Request blocked as a precaution."
            ),
            "status": "ERROR",
            "elapsed_time": (
                time.perf_counter() - start
            ),
            "openai_calls": 1,
            "cache_hit": False,
            "cache_match": None,
            "error": str(error),
        }


# ============================================================
# COMMON SECURITY GATE
# ============================================================

def run_security_gate(query: str):
    start = time.perf_counter()

    deterministic = (
        run_deterministic_guardrails(query)
    )

    if not deterministic["allowed"]:
        return {
            "allowed": False,
            "deterministic": deterministic,
            "ai_safety": None,
            "elapsed_time": (
                time.perf_counter() - start
            ),
            "openai_calls": 0,
        }

    ai_safety = run_ai_safety_guardrail(
        deterministic["query"]
    )

    return {
        "allowed": ai_safety["allowed"],
        "deterministic": deterministic,
        "ai_safety": ai_safety,
        "elapsed_time": (
            time.perf_counter() - start
        ),
        "openai_calls": ai_safety.get("openai_calls", 1),
        "cache_hits": 1 if ai_safety.get("cache_hit") else 0,
    }


# ============================================================
# NON-AGENTIC ENGINE
# SAME BUSINESS CAPABILITIES WITHOUT AGENTS
# 2 DIRECT OPENAI CALLS + 1 WORKFLOW SERPER CALL
# RECORD BUILD / VALIDATION / PERSISTENCE ARE DETERMINISTIC PYTHON
# ============================================================

def run_non_agentic_web_search(query: str, session_memory: str = ""):
    """Retrieve web evidence with one Serper call when cache reuse is unavailable.

    Exact and very-near duplicate search requests can reuse session evidence.
    Reuse is date-scoped and conservative so freshness-sensitive queries do not
    carry stale evidence across days.
    """
    # Include compact conversational context in the cache identity so a
    # follow-up question cannot accidentally reuse evidence from the wrong topic.
    cache_query = (
        query
        if not session_memory
        else f"{query}\nSESSION CONTEXT:\n{session_memory}"
    )

    cached, cache_match = _cache_lookup(
        "workflow_search_cache",
        cache_query,
        allow_near=True,
        threshold=0.96,
    )
    if cached is not None:
        cached["elapsed_time"] = 0.0
        cached["serper_calls"] = 0
        cached["cache_hit"] = True
        cached["cache_match"] = cache_match
        return cached

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }

    freshness_terms = (
        "latest", "current", "newest", "today",
        "recent", "now", "stable version",
        "next", "upcoming", "nearest", "when is",
    )
    lower_query = query.lower()

    search_query = query
    if session_memory:
        search_query = (
            f"Conversation context: {session_memory}. "
            f"Current customer request: {query}"
        )

    if any(term in lower_query for term in freshness_terms):
        date_ctx = current_date_context()
        search_query = (
            f"{search_query} official authoritative information. "
            f"Current date is {date_ctx['readable']} ({date_ctx['iso']}). "
            "For next/upcoming/nearest requests, return the nearest valid future "
            "result on or after the current date. "
            "For latest/current/newest requests, return the most current result."
        )

    payload = {"q": search_query, "num": 6}
    start = time.perf_counter()

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        items = []
        answer_box = data.get("answerBox")
        if answer_box:
            items.append({
                "title": answer_box.get("title", "Answer Box"),
                "snippet": (
                    answer_box.get("answer")
                    or answer_box.get("snippet")
                    or ""
                ),
                "link": answer_box.get("link", ""),
            })

        for item in data.get("organic", [])[:5]:
            items.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", ""),
            })

        output = {
            "status": "SUCCESS",
            "items": items,
            "elapsed_time": time.perf_counter() - start,
            "serper_calls": 1,
            "search_query": search_query,
            "cache_hit": False,
            "cache_match": None,
        }
        _cache_store("workflow_search_cache", cache_query, output)
        return output
    except Exception as error:
        return {
            "status": "ERROR",
            "items": [],
            "elapsed_time": time.perf_counter() - start,
            "serper_calls": 1,
            "search_query": search_query,
            "cache_hit": False,
            "cache_match": None,
            "error": str(error),
        }


def run_non_agentic(query: str, workflow_search=None):
    """Execute the same support workflow without an agent framework.

    Call 1: direct assistant answer from model knowledge.
    Tool call: Python invokes Serper directly.
    Call 2: direct model synthesis using retrieved web evidence.
    Python then builds, validates and persists the support record.
    """
    total_start = time.perf_counter()

    assistant_system_prompt = """
You are a professional customer-support assistant.
Answer the user's question clearly and accurately using only your existing
model knowledge. Do not claim that you searched or verified information online.
"""

    first_start = time.perf_counter()
    first_response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": assistant_system_prompt},
            {"role": "user", "content": query},
        ],
    )
    first_elapsed = time.perf_counter() - first_start
    assistant_answer = first_response.choices[0].message.content.strip()

    # Reuse a caller-supplied workflow search when available. This lets
    # Experiment 1 compare both architectures against the SAME retrieved
    # evidence with only one external workflow-search call.
    web_search = workflow_search or run_non_agentic_web_search(query)
    web_evidence = format_evidence(web_search)
    date_ctx = current_date_context()

    grounded_prompt = f"""
You are a professional customer-support research assistant.
Answer the customer query using the supplied web evidence.
Prefer evidence from authoritative sources when available.
Do not invent facts that are not supported by the evidence.
If the evidence is insufficient, say so clearly.

CURRENT DATE:
{date_ctx["readable"]} ({date_ctx["iso"]})

CUSTOMER QUERY:
{query}

INITIAL ASSISTANT ANSWER:
{assistant_answer}

WEB EVIDENCE:
{web_evidence}

For next/upcoming/nearest date requests, never present a date earlier than
CURRENT DATE as the selected answer. For latest/current/newest requests, use
the most current item supported by the evidence.
"""

    second_start = time.perf_counter()
    second_response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[{"role": "user", "content": grounded_prompt}],
    )
    second_elapsed = time.perf_counter() - second_start
    web_answer = second_response.choices[0].message.content.strip()

    temporal_validation = run_temporal_validation(
        query,
        web_answer,
    )

    entry_record = (
        "CUSTOMER QUERY\n"
        f"{query.strip()}\n\n"
        "ASSISTANT ANSWER\n"
        f"{assistant_answer}\n\n"
        "WEB SEARCH ANSWER\n"
        f"{web_answer}"
    )

    validation = validate_agentic_output(
        assistant_answer,
        web_answer,
        entry_record,
    )

    saved_file = None
    if validation["passed"]:
        saved_file = save_agentic_record(entry_record)

    first_usage = first_response.usage
    second_usage = second_response.usage
    total_tokens = None
    if first_usage and second_usage:
        total_tokens = first_usage.total_tokens + second_usage.total_tokens

    return {
        "architecture": "Non-Agentic",
        "assistant_answer": assistant_answer,
        "answer": web_answer,
        "web_answer": web_answer,
        "entry_record": entry_record_display,
        "output_guardrail": output_guardrail,
        "elapsed_time": time.perf_counter() - total_start,
        "assistant_elapsed_time": first_elapsed,
        "web_generation_elapsed_time": second_elapsed,
        "agents": 0,
        "web_search": True,
        "workflow_search": web_search,
        "workflow_serper_calls": web_search.get("serper_calls", 0),
        "orchestration": "Python",
        "persistence": bool(saved_file),
        "saved_file": saved_file,
        "output_validation": validation,
        "temporal_validation": temporal_validation,
        "current_date": date_ctx["iso"],
        "known_openai_calls": 2,
        "total_tokens": total_tokens,
    }


# ============================================================
# AGENTIC CREW
# CREWAI CONTROLS MODEL INVOCATIONS
# ============================================================

def create_support_crew(web_evidence_text: str):
    """Create the required three-agent workflow using one cached evidence set."""

    @tool("Workflow Web Search Evidence")
    def workflow_web_search_evidence(search_query: str) -> str:
        """Return the web evidence already retrieved once for this workflow."""
        return web_evidence_text

    assistant_agent = Agent(
        role="Assistant",
        goal="Provide a clear and helpful response using existing model knowledge.",
        backstory=(
            "You are the first customer-support agent. "
            "You answer without web search."
        ),
        verbose=False,
        allow_delegation=False,
    )

    web_agent = Agent(
        role="Web Search Assistant",
        goal=(
            "Use supplied current web-search evidence to produce an accurate, "
            "evidence-informed customer-support answer."
        ),
        backstory=(
            "You are a customer-support research specialist. "
            "The application has already retrieved the web evidence once, "
            "so reuse it rather than triggering another external search."
        ),
        tools=[workflow_web_search_evidence],
        verbose=False,
        allow_delegation=False,
    )

    entry_agent = Agent(
        role="Entry Agent",
        goal=(
            "Prepare a complete support record containing the original query "
            "and both previous answers."
        ),
        backstory=(
            "You maintain accurate customer-support records without changing "
            "previous agents' meaning."
        ),
        verbose=False,
        allow_delegation=False,
    )

    assistant_task = Task(
        description=(
            "Answer this customer query using your existing model knowledge only.\\n\\n"
            "CUSTOMER QUERY:\\n{query}\\n\\n"
            "SESSION MEMORY FROM EARLIER TURNS:\\n{session_memory}\\n\\n"
            "Use session memory to resolve follow-up references such as 'it', "
            "'that version', 'that product', 'the previous one' or similar. "
            "The current query has priority. Do not invent context that is not "
            "present in the session memory.\\n\\n"
            "If the user asks for a stable release/version, do not describe an "
            "alpha, beta, release candidate or other prerelease as stable.\\n\\n"
            "Do not claim that you searched the web."
        ),
        expected_output="A clear customer-support response.",
        agent=assistant_agent,
    )

    web_task = Task(
        description=(
            "Produce the current, evidence-informed answer for this customer query.\\n\\n"
            "CUSTOMER QUERY:\\n{query}\\n\\n"
            "SESSION MEMORY FROM EARLIER TURNS:\\n{session_memory}\\n\\n"
            "CURRENT DATE:\\n{current_date}\\n\\n"
            "Use session memory to resolve pronouns and follow-up references before "
            "interpreting the current request. The current query and current web "
            "evidence override older session context.\\n\\n"
            "The first agent response is available as context.\\n\\n"
            "CURRENT WEB SEARCH EVIDENCE ALREADY RETRIEVED BY THE APPLICATION:\\n"
            "{web_evidence}\\n\\n"
            "Instructions:\\n"
            "1. Use the supplied evidence as the factual basis for current claims.\\n"
            "2. Prefer official/authoritative evidence when present.\\n"
            "3. For latest/current/newest questions, identify the newest item "
            "supported by the evidence. If the query explicitly asks for the "
            "latest stable release/version, exclude alpha, beta, release-candidate "
            "and other prerelease builds.\\n"
            "4. For next/upcoming/nearest date questions, discard dates earlier "
            "than CURRENT DATE and select the nearest valid future date.\\n"
            "5. Never describe a date earlier than CURRENT DATE as next or upcoming.\\n"
            "6. Do not fall back to older model-memory facts when evidence is newer.\\n"
            "7. Do not invent unsupported details.\\n"
            "8. The Workflow Web Search Evidence tool returns the same cached "
            "evidence and makes zero external search calls."
        ),
        expected_output=(
            "A current customer-support response grounded in supplied web evidence."
        ),
        agent=web_agent,
        context=[assistant_task],
    )

    entry_task = Task(
        description=(
            "Create ONE final grounded customer-support answer by analyzing "
            "the two previous outputs.\\n\\n"
            "CUSTOMER QUERY:\\n{query}\\n\\n"
            "SESSION MEMORY FROM EARLIER TURNS:\\n{session_memory}\\n\\n"
            "CURRENT DATE:\\n{current_date}\\n\\n"
            "Use session memory to resolve what follow-up references mean. Current "
            "web evidence and the current customer request override older context.\\n\\n"
            "The Assistant Agent answer and Web Search Assistant answer are "
            "available as context.\\n\\n"
            "Rules:\\n"
            "1. Reconcile the two answers instead of repeating them.\\n"
            "2. For factual/current/latest information, prefer the Web Search "
            "Assistant when it is supported by the retrieved evidence. If the "
            "customer explicitly asks for a stable version/release, never return "
            "an alpha, beta or release candidate as the stable answer.\\n"
            "3. For next/upcoming/nearest date requests, never present a date "
            "earlier than CURRENT DATE as the answer. Select the nearest valid "
            "future date supported by the Web Search Assistant evidence.\\n"
            "4. Put the actual selected future date first in the answer.\\n"
            "5. Keep useful explanatory detail from the Assistant Agent only "
            "when it does not conflict with the evidence-informed answer.\\n"
            "6. Remove duplicated information.\\n"
            "7. Do not mention internal agents, orchestration or evaluation.\\n"
            "8. Do not invent unsupported facts.\\n"
            "9. Return the final customer-facing answer with no headings.\\n"
            "10. After the answer, append exactly one internal metadata line: "
            "[[SELECTED_EVENT_DATE: YYYY-MM-DD]]. For next/upcoming/nearest "
            "event questions, choose the actual event/observance date the user "
            "is asking for, not a start-time or end-time boundary. Example: if "
            "a fast begins the prior evening but is observed on September 22, "
            "use 2026-09-22. For non-relative-date questions append "
            "[[SELECTED_EVENT_DATE: NONE]]."
        ),
        expected_output=(
            "One concise, accurate, evidence-grounded final customer answer, "
            "followed by one SELECTED_EVENT_DATE metadata marker."
        ),
        agent=entry_agent,
        context=[assistant_task, web_task],
    )

    crew = Crew(
        agents=[assistant_agent, web_agent, entry_agent],
        tasks=[assistant_task, web_task, entry_task],
        process=Process.sequential,
        verbose=False,
    )

    return crew, assistant_task, web_task, entry_task


def validate_agentic_output(
    assistant_answer,
    web_answer,
    final_answer_or_record,
    record=None,
):
    """
    Validate both workflow variants without requiring another model call.

    - Non-agentic path passes its legacy three-part record as the third argument.
    - Agentic path passes final grounded answer + persisted record.
    """
    if record is None:
        legacy_record = final_answer_or_record
        checks = {
            "assistant_output": bool(assistant_answer.strip()),
            "web_output": bool(web_answer.strip()),
            "entry_record": (
                bool(legacy_record.strip())
                and "CUSTOMER QUERY" in legacy_record.upper()
                and "ASSISTANT ANSWER" in legacy_record.upper()
                and "WEB SEARCH ANSWER" in legacy_record.upper()
            ),
        }
    else:
        final_answer = final_answer_or_record
        checks = {
            "assistant_output": bool(assistant_answer.strip()),
            "web_output": bool(web_answer.strip()),
            "final_grounded_answer": bool(final_answer.strip()),
            "entry_record": (
                bool(record.strip())
                and "CUSTOMER QUERY" in record.upper()
                and "FINAL GROUNDED ANSWER" in record.upper()
            ),
        }

    return {
        "passed": all(checks.values()),
        "checks": checks,
    }



# ============================================================
# OUTPUT GUARDRAILS
# Deterministic post-generation gate — zero additional API calls.
# Runs BEFORE persistence / user display of the final answer.
# ============================================================

def run_output_guardrail(
    query: str,
    assistant_answer: str,
    web_answer: str,
    final_answer: str,
    temporal_validation=None,
):
    """
    Deterministic output safety gate.

    Checks:
    - required generated outputs are present and reasonably bounded
    - no obvious secret / credential leakage
    - no internal prompt / metadata leakage
    - no unsupported claims that the Buildathon performed real account actions
    - deterministic temporal validation, when applicable

    This is intentionally zero-call. Factual answer quality remains the job of
    grounding/evaluation; this gate focuses on safe release of generated output.
    """
    outputs = {
        "Assistant Agent": (assistant_answer or "").strip(),
        "Web Search Assistant": (web_answer or "").strip(),
        "Final Answer": (final_answer or "").strip(),
    }

    combined = "\n\n".join(
        f"{name}: {value}"
        for name, value in outputs.items()
    )

    checks = []

    def add_check(name, passed, detail):
        checks.append({
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    # 1. Required output presence / reasonable size.
    presence_ok = all(bool(value) for value in outputs.values())
    final_length_ok = 20 <= len(outputs["Final Answer"]) <= 12000
    add_check(
        "Output structure",
        presence_ok and final_length_ok,
        (
            "Required agent outputs are present and the final answer length is within limits."
            if presence_ok and final_length_ok
            else "A required output is empty or the final answer length is outside allowed limits."
        ),
    )

    # 2. Credential / secret leakage patterns.
    secret_patterns = [
        r"\bsk-[A-Za-z0-9_-]{16,}\b",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"(?i)\b(?:api[_ -]?key|access[_ -]?token|secret[_ -]?key|password)\s*[:=]\s*[\"']?[A-Za-z0-9_\-\/+=.]{8,}",
    ]
    leaked_secret = next(
        (
            pattern
            for pattern in secret_patterns
            if re.search(pattern, combined)
        ),
        None,
    )
    add_check(
        "Secrets leakage",
        leaked_secret is None,
        (
            "No credential or secret-like value was detected in generated output."
            if leaked_secret is None
            else "A secret-like value was detected; output release is blocked."
        ),
    )

    # 3. Internal instruction / metadata leakage.
    internal_markers = [
        "SESSION MEMORY FROM EARLIER TURNS",
        "RAW WEB SEARCH EVIDENCE",
        "SYSTEM PROMPT",
        "DEVELOPER MESSAGE",
        "[[SELECTED_EVENT_DATE:",
        "CURRENT WEB SEARCH EVIDENCE ALREADY RETRIEVED BY THE APPLICATION",
    ]
    leaked_marker = next(
        (
            marker
            for marker in internal_markers
            if marker.lower() in combined.lower()
        ),
        None,
    )
    add_check(
        "Internal-context leakage",
        leaked_marker is None,
        (
            "No internal prompt, memory or hidden metadata marker was exposed."
            if leaked_marker is None
            else f"Internal marker detected in generated output: {leaked_marker}"
        ),
    )

    # 4. Buildathon has no real account-operation tools. Do not let the final
    #    answer claim that an external action was actually completed.
    unsupported_action_patterns = [
        r"(?i)\b(?:i|we)(?:'ve| have)\s+(?:successfully\s+)?refunded\s+(?:your|the)\b",
        r"(?i)\b(?:i|we)(?:'ve| have)\s+(?:successfully\s+)?reset\s+(?:your|the)\s+(?:password|account)\b",
        r"(?i)\b(?:i|we)(?:'ve| have)\s+(?:successfully\s+)?unlocked\s+(?:your|the)\s+account\b",
        r"(?i)\b(?:i|we)(?:'ve| have)\s+(?:successfully\s+)?created\s+(?:a|the)\s+(?:ticket|case)\b",
        r"(?i)\b(?:i|we)(?:'ve| have)\s+(?:successfully\s+)?escalated\s+(?:your|the)\s+(?:ticket|case|request)\b",
        r"(?i)\b(?:i|we)(?:'ve| have)\s+(?:successfully\s+)?cancel(?:led|ed)\s+(?:your|the)\s+(?:order|subscription)\b",
    ]
    unsupported_action = next(
        (
            pattern
            for pattern in unsupported_action_patterns
            if re.search(pattern, outputs["Final Answer"])
        ),
        None,
    )
    add_check(
        "Unsupported action claims",
        unsupported_action is None,
        (
            "The final answer does not claim a real external action that this Buildathon cannot perform."
            if unsupported_action is None
            else "The final answer claims a real-world support action without an execution tool."
        ),
    )

    # 5. Reuse the deterministic temporal validation already computed.
    temporal_validation = temporal_validation or {}
    temporal_applied = bool(temporal_validation.get("applied"))
    temporal_ok = (
        True
        if not temporal_applied
        else bool(temporal_validation.get("passed"))
    )
    add_check(
        "Temporal safety",
        temporal_ok,
        (
            temporal_validation.get("detail", "Temporal validation passed.")
            if temporal_applied
            else "No temporal next/upcoming validation was required for this response."
        ),
    )

    passed = all(check["status"] == "PASS" for check in checks)

    return {
        "passed": passed,
        "status": "PASS" if passed else "BLOCKED",
        "checks": checks,
        "llm_calls": 0,
        "search_calls": 0,
    }


def save_agentic_record(record: str):
    file_name = "answers.txt"

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
        file_name,
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n"
            + "=" * 70
            + "\n"
        )

        file.write(
            f"Timestamp: {timestamp}\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            record.strip()
        )

        file.write("\n\n")

    return file_name


def parse_entry_agent_output(raw_output: str):
    """Strip internal temporal metadata from the Entry Agent output."""
    raw_output = (raw_output or "").strip()

    marker_pattern = re.compile(
        r"\n?\[\[SELECTED_EVENT_DATE:\s*(\d{4}-\d{2}-\d{2}|NONE)\s*\]\]\s*$",
        flags=re.IGNORECASE,
    )

    match = marker_pattern.search(raw_output)

    if not match:
        return raw_output, None

    value = match.group(1).upper()
    clean_answer = raw_output[:match.start()].strip()

    return clean_answer, (None if value == "NONE" else value)


def run_agentic(
    query: str,
    workflow_search=None,
    session_memory: str = "",
    search_memory: str = "",
    progress_callback=None,
):
    """Run three agents with exactly one external workflow search."""
    total_start = time.perf_counter()

    # Reuse shared workflow evidence in controlled comparisons.
    workflow_search = workflow_search or run_non_agentic_web_search(query, session_memory=search_memory)
    web_evidence_text = format_evidence(workflow_search)

    crew, assistant_task, web_task, entry_task = create_support_crew(
        web_evidence_text
    )

    crew_start = time.perf_counter()
    date_ctx = current_date_context()

    crew.kickoff(
        inputs={
            "query": query,
            "session_memory": session_memory or "No earlier session context.",
            "web_evidence": web_evidence_text,
            "current_date": date_ctx["readable"],
        }
    )
    crew_elapsed = time.perf_counter() - crew_start

    assistant_answer = assistant_task.output.raw.strip()
    web_answer = web_task.output.raw.strip()
    raw_entry_output = entry_task.output.raw.strip()
    final_answer, selected_event_date = parse_entry_agent_output(
        raw_entry_output
    )

    temporal_validation = run_temporal_validation(
        query,
        final_answer,
        selected_event_date=selected_event_date,
    )

    if progress_callback:
        progress_callback(
            "output_guardrail",
            "Checking generated output before persistence: leakage, internal "
            "context, unsupported actions and temporal safety.",
        )

    output_guardrail = run_output_guardrail(
        query=query,
        assistant_answer=assistant_answer,
        web_answer=web_answer,
        final_answer=final_answer,
        temporal_validation=temporal_validation,
    )

    # Prepare a clean record. Persistence is allowed only after BOTH structural
    # validation and output guardrails pass.
    entry_record = (
        "CUSTOMER QUERY\\n"
        f"{query.strip()}\\n\\n"
        "FINAL GROUNDED ANSWER\\n"
        f"{final_answer}"
    )

    validation = validate_agentic_output(
        assistant_answer,
        web_answer,
        final_answer,
        entry_record,
    )
    validation["checks"]["output_guardrail"] = output_guardrail["passed"]
    validation["passed"] = bool(
        validation["passed"]
        and output_guardrail["passed"]
    )

    saved_file = None
    if validation["passed"]:
        saved_file = save_agentic_record(entry_record)

    # If the post-generation gate blocks the answer, do not release raw
    # intermediate/final generated text to the UI and do not persist it.
    if not output_guardrail["passed"]:
        blocked_message = (
            "The generated response was blocked by the output guardrail and "
            "was not persisted. Review the Output Guardrail evidence below."
        )
        assistant_display = blocked_message
        web_display = blocked_message
        final_display = blocked_message
        entry_record_display = (
            "CUSTOMER QUERY\\n"
            f"{query.strip()}\\n\\n"
            "FINAL GROUNDED ANSWER\\n"
            f"{blocked_message}"
        )
    else:
        assistant_display = assistant_answer
        web_display = web_answer
        final_display = final_answer
        entry_record_display = entry_record

    return {
        "architecture": "Agentic",
        "assistant_answer": assistant_display,
        "answer": final_display,
        "final_answer": final_display,
        "web_answer": web_display,
        "entry_record": entry_record_display,
        "output_guardrail": output_guardrail,
        "elapsed_time": time.perf_counter() - total_start,
        "crew_elapsed_time": crew_elapsed,
        "agents": 3,
        "web_search": True,
        "workflow_search": workflow_search,
        "workflow_serper_calls": workflow_search.get("serper_calls", 0),
        "workflow_search_query": workflow_search.get("search_query", query),
        "orchestration": True,
        "process": "Sequential",
        "persistence": bool(saved_file),
        "saved_file": saved_file,
        "output_validation": validation,
        "temporal_validation": temporal_validation,
        "selected_event_date": selected_event_date,
        "current_date": date_ctx["iso"],
        "known_openai_calls": None,
    }


def evaluate_buildathon_from_existing_output(query: str, core_result: dict):
    """Evaluate using the workflow's existing raw search evidence.

    Cost profile:
    - 0 additional Serper/search calls
    - 1 evaluator OpenAI call
    """
    final_answer = core_result.get("final_answer") or core_result.get("answer", "")
    workflow_search = core_result.get("workflow_search") or {}
    evidence_text = format_evidence(workflow_search)
    temporal_validation = (
        core_result.get("temporal_validation")
        or run_temporal_validation(
            query,
            final_answer,
            selected_event_date=core_result.get("selected_event_date"),
        )
    )
    date_ctx = current_date_context()

    prompt = f"""
You are evaluating a completed three-agent customer-support workflow.

CURRENT DATE:
{date_ctx["readable"]} ({date_ctx["iso"]})

CUSTOMER QUERY:
{query}

FINAL GROUNDED ANSWER:
{final_answer}

RAW WEB SEARCH EVIDENCE ALREADY USED BY THE WORKFLOW:
{evidence_text}

Do not perform or request another web search.

Evaluate:

RELEVANCE:
How directly the answer addresses the query.

COMPLETENESS:
Whether important information supported by evidence is included.

CONSISTENCY:
Whether the answer is internally coherent and non-contradictory.

GROUNDEDNESS:
Whether factual/current claims in the FINAL GROUNDED ANSWER are supported by the RAW WEB SEARCH EVIDENCE.
Penalize outdated or unsupported "latest/current" claims.
For next/upcoming/nearest requests, a claimed date earlier than CURRENT DATE is incorrect.

Return ONLY valid JSON:
{{
  "relevance": {{"score": 0, "reason": ""}},
  "completeness": {{"score": 0, "reason": ""}},
  "consistency": {{"score": 0, "reason": ""}},
  "groundedness": {{"score": 0, "reason": ""}}
}}

All scores must be integers from 0 to 100.
Keep reasons concise.
"""
    start_time = time.perf_counter()

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        result = parse_json_response(
            response.choices[0].message.content
        )

        for metric in [
            "relevance",
            "completeness",
            "consistency",
            "groundedness",
        ]:
            item = result.get(metric) or {}
            result[metric] = {
                "score": clamp_score(item.get("score", 0)),
                "reason": str(item.get("reason", "")).strip(),
            }

        if (
            temporal_validation.get("applied")
            and not temporal_validation.get("passed")
        ):
            groundedness = result.get("groundedness") or {}
            groundedness["score"] = min(
                clamp_score(groundedness.get("score", 0)),
                40,
            )
            existing_reason = str(
                groundedness.get("reason", "")
            ).strip()
            temporal_reason = temporal_validation.get(
                "detail",
                "Temporal validation failed.",
            )
            groundedness["reason"] = (
                f"{existing_reason} Deterministic temporal check: "
                f"{temporal_reason}"
            ).strip()
            result["groundedness"] = groundedness

        return {
            "status": "SUCCESS",
            "result": {"agentic": result},
            "temporal_validation": temporal_validation,
            "elapsed_time": time.perf_counter() - start_time,
            "openai_calls": 1,
            "serper_calls": 0,
            "evidence_source": "Reused workflow raw search evidence",
        }

    except Exception as error:
        return {
            "status": "ERROR",
            "result": {},
            "elapsed_time": time.perf_counter() - start_time,
            "openai_calls": 1,
            "serper_calls": 0,
            "error": str(error),
        }


# ============================================================
# INDEPENDENT EVALUATION EVIDENCE
# ONE SERPER CALL PER SUCCESSFUL RUN
# ============================================================

def retrieve_evaluation_evidence(
    query: str,
):
    """Retrieve independent evaluation evidence, with session reuse.

    This cache is intentionally separate from workflow evidence so Experiment 1
    still has an independent verification source. Repeated or near-identical
    evaluation questions can reuse the already retrieved independent evidence.
    """
    cached, cache_match = _cache_lookup(
        "evaluation_search_cache",
        query,
        allow_near=True,
        threshold=0.96,
    )
    if cached is not None:
        cached["elapsed_time"] = 0.0
        cached["serper_calls"] = 0
        cached["cache_hit"] = True
        cached["cache_match"] = cache_match
        return cached

    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }

    freshness_terms = (
        "latest", "current", "newest", "today",
        "recent", "now", "stable version",
        "next", "upcoming", "nearest", "when is",
    )
    eval_query = query
    if any(term in query.lower() for term in freshness_terms):
        date_ctx = current_date_context()
        eval_query = (
            f"{query} independent verification authoritative sources. "
            f"Current date is {date_ctx['readable']} ({date_ctx['iso']})."
        )

    payload = {
        "q": eval_query,
        "num": 5,
    }

    start = time.perf_counter()

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=15,
        )

        response.raise_for_status()
        data = response.json()
        items = []

        answer_box = data.get("answerBox")
        if answer_box:
            items.append({
                "title": answer_box.get("title", "Answer Box"),
                "snippet": (
                    answer_box.get("answer")
                    or answer_box.get("snippet")
                    or ""
                ),
                "link": answer_box.get("link", ""),
            })

        knowledge_graph = data.get("knowledgeGraph")
        if knowledge_graph:
            items.append({
                "title": knowledge_graph.get("title", "Knowledge Graph"),
                "snippet": knowledge_graph.get("description", ""),
                "link": knowledge_graph.get("descriptionLink", ""),
            })

        for item in data.get("organic", [])[:5]:
            items.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", ""),
            })

        output = {
            "status": "SUCCESS",
            "items": items,
            "elapsed_time": time.perf_counter() - start,
            "serper_calls": 1,
            "search_query": eval_query,
            "cache_hit": False,
            "cache_match": None,
        }
        _cache_store("evaluation_search_cache", query, output)
        return output

    except Exception as error:
        return {
            "status": "ERROR",
            "items": [],
            "elapsed_time": time.perf_counter() - start,
            "serper_calls": 1,
            "search_query": eval_query,
            "cache_hit": False,
            "cache_match": None,
            "error": str(error),
        }


def format_evidence(evidence):
    items = evidence.get(
        "items",
        [],
    )

    if not items:
        return (
            "NO EXTERNAL EVIDENCE AVAILABLE"
        )

    blocks = []

    for index, item in enumerate(
        items,
        start=1,
    ):
        blocks.append(
            f"""
SOURCE {index}
Title: {item["title"]}
Snippet: {item["snippet"]}
URL: {item["link"]}
""".strip()
        )

    return "\n\n".join(blocks)


# ============================================================
# SINGLE COMMON EVALUATOR
# ONE OPENAI CALL TOTAL
# ============================================================

def run_common_evaluation(
    query,
    evidence,
    non_agentic=None,
    agentic=None,
):
    evidence_text = (
        format_evidence(evidence)
    )

    non_answer = (
        non_agentic["answer"]
        if non_agentic
        else "NOT EXECUTED"
    )

    agent_answer = (
        agentic["answer"]
        if agentic
        else "NOT EXECUTED"
    )

    compare_both = (
        non_agentic is not None
        and agentic is not None
    )

    prompt = f"""
You are the evaluation layer for an AI
customer-support learning application.

The generator architectures have already completed.
You are an independent evaluator.

CUSTOMER QUERY:

{query}


NON-AGENTIC ANSWER:

{non_answer}


AGENTIC WEB-ASSISTED ANSWER:

{agent_answer}


INDEPENDENT EXTERNAL EVIDENCE:

{evidence_text}


Evaluate only architectures that were actually executed.

Return ONLY valid JSON in this exact structure:

{{
  "non_agentic": {{
    "executed": true,
    "relevance": {{
      "score": 0,
      "reason": ""
    }},
    "completeness": {{
      "score": 0,
      "reason": ""
    }},
    "consistency": {{
      "score": 0,
      "reason": ""
    }},
    "groundedness": {{
      "score": 0,
      "reason": ""
    }}
  }},
  "agentic": {{
    "executed": true,
    "relevance": {{
      "score": 0,
      "reason": ""
    }},
    "completeness": {{
      "score": 0,
      "reason": ""
    }},
    "consistency": {{
      "score": 0,
      "reason": ""
    }},
    "groundedness": {{
      "score": 0,
      "reason": ""
    }}
  }},
  "comparison": {{
    "available": true,
    "agreement": "AGREE",
    "reason": ""
  }}
}}

Scoring:

RELEVANCE:
How directly the answer addresses the query.

COMPLETENESS:
Whether important information needed to answer
the query is included.

CONSISTENCY:
Whether the answer is internally coherent and
does not contradict itself.

GROUNDEDNESS:
Whether factual claims are supported by the
independent evidence supplied above.

The independent evidence is evaluation-time evidence.
Do NOT assume that the non-agentic generator had access
to it.

Comparison classifications:

AGREE:
Important claims materially align.

PARTIAL:
There are meaningful differences or additions,
but no major direct factual conflict.

DISAGREE:
Important factual claims materially conflict.

If an architecture says NOT EXECUTED:

- Set its executed field to false.
- Return score 0 for its metrics.
- Explain that it was not executed.
- Do not evaluate it.

Comparison available must be:
{str(compare_both).lower()}

If comparison is unavailable:
- set available to false
- set agreement to "NOT_APPLICABLE"
- explain that only one architecture ran.

All scores must be integer values from 0 to 100.

Keep reasons concise and evidence-focused.
"""

    start = time.perf_counter()

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        result = parse_json_response(
            response.choices[0].message.content
        )

        for architecture in [
            "non_agentic",
            "agentic",
        ]:
            architecture_result = (
                result.get(
                    architecture,
                    {},
                )
            )

            for metric in [
                "relevance",
                "completeness",
                "consistency",
                "groundedness",
            ]:
                metric_result = (
                    architecture_result.get(
                        metric,
                        {},
                    )
                )

                metric_result[
                    "score"
                ] = clamp_score(
                    metric_result.get(
                        "score",
                        0,
                    )
                )

                metric_result[
                    "reason"
                ] = str(
                    metric_result.get(
                        "reason",
                        "",
                    )
                ).strip()

                architecture_result[
                    metric
                ] = metric_result

            result[
                architecture
            ] = architecture_result

        comparison = result.get(
            "comparison",
            {},
        )

        agreement = str(
            comparison.get(
                "agreement",
                "NOT_APPLICABLE",
            )
        ).strip().upper()

        allowed_agreements = {
            "AGREE",
            "PARTIAL",
            "DISAGREE",
            "NOT_APPLICABLE",
        }

        if agreement not in allowed_agreements:
            agreement = (
                "PARTIAL"
                if compare_both
                else "NOT_APPLICABLE"
            )

        comparison[
            "agreement"
        ] = agreement

        comparison[
            "available"
        ] = compare_both

        comparison[
            "reason"
        ] = str(
            comparison.get(
                "reason",
                "",
            )
        ).strip()

        result[
            "comparison"
        ] = comparison

        usage = response.usage

        return {
            "status": "SUCCESS",
            "result": result,
            "elapsed_time": (
                time.perf_counter()
                - start
            ),
            "openai_calls": 1,
            "total_tokens": (
                usage.total_tokens
                if usage
                else None
            ),
        }

    except Exception as error:
        return {
            "status": "ERROR",
            "result": None,
            "elapsed_time": (
                time.perf_counter()
                - start
            ),
            "openai_calls": 1,
            "error": str(error),
        }


# ============================================================
# DETERMINISTIC OBSERVATION ENGINE
# NO OPENAI CALL
# ============================================================

def build_observations(
    non_agentic,
    agentic,
    evaluation,
):
    observations = []

    if (
        non_agentic
        and agentic
    ):
        non_time = (
            non_agentic[
                "elapsed_time"
            ]
        )

        agent_time = (
            agentic[
                "elapsed_time"
            ]
        )

        difference = (
            agent_time
            - non_time
        )

        observations.append(
            {
                "title": "Latency",
                "text": (
                    f"Non-agentic generation completed "
                    f"in {non_time:.2f}s. "
                    f"Agentic generation completed "
                    f"in {agent_time:.2f}s. "
                    f"Observed difference: "
                    f"{difference:+.2f}s."
                ),
            }
        )

        observations.append(
            {
                "title": "Architecture",
                "text": (
                    "The non-agentic path used explicit Python "
                    "orchestration with zero agents: direct model "
                    "calls, direct web search, deterministic record "
                    "construction, validation and persistence. "
                    "The agentic path used three "
                    "specialized CrewAI agents in a "
                    "sequential process, including "
                    "web search and record persistence."
                ),
            }
        )

    elif non_agentic:
        observations.append(
            {
                "title": "Architecture",
                "text": (
                    "This run used the non-agentic architecture: "
                    "zero agents, explicit Python orchestration, "
                    "direct model calls, direct web search, and "
                    "deterministic record persistence."
                ),
            }
        )

    elif agentic:
        observations.append(
            {
                "title": "Architecture",
                "text": (
                    "This run used the agentic "
                    "architecture: three CrewAI agents "
                    "running sequentially with web search "
                    "and Entry Agent persistence."
                ),
            }
        )

    if (
        evaluation
        and evaluation["status"]
        == "SUCCESS"
    ):
        result = (
            evaluation["result"]
        )

        comparison = result.get(
            "comparison",
            {},
        )

        if comparison.get(
            "available"
        ):
            observations.append(
                {
                    "title": "Answer Relationship",
                    "text": (
                        f"The evaluator classified "
                        f"the two answers as "
                        f"{comparison['agreement']}. "
                        f"{comparison['reason']}"
                    ),
                }
            )

        if (
            non_agentic
            and agentic
        ):
            non_ground = (
                result[
                    "non_agentic"
                ][
                    "groundedness"
                ][
                    "score"
                ]
            )

            agent_ground = (
                result[
                    "agentic"
                ][
                    "groundedness"
                ][
                    "score"
                ]
            )

            observations.append(
                {
                    "title": "Evidence Alignment",
                    "text": (
                        f"Against the same independent "
                        f"evaluation evidence, "
                        f"non-agentic groundedness was "
                        f"{non_ground}/100 and agentic "
                        f"groundedness was "
                        f"{agent_ground}/100."
                    ),
                }
            )

    return observations




# ============================================================
# BUILDATHON — DETERMINISTIC VALIDATION EVIDENCE
# ZERO ADDITIONAL MODEL / SEARCH CALLS
# ============================================================

def build_buildathon_validation(core_result, security, evaluation):
    """Validate the Buildathon execution using existing measured run state only."""
    checks = []

    def add_check(name, passed, detail):
        checks.append({
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    guardrails_passed = bool(security and security.get("allowed"))
    add_check(
        "Security gate",
        guardrails_passed,
        "Deterministic guardrails and AI safety allowed the request."
        if guardrails_passed
        else "The request did not pass the complete security gate.",
    )

    sequence_passed = (
        core_result.get("agents") == 3
        and core_result.get("process") == "Sequential"
        and bool(core_result.get("assistant_answer"))
        and bool(core_result.get("web_answer"))
        and bool(core_result.get("final_answer") or core_result.get("answer"))
    )
    add_check(
        "Three-agent sequence",
        sequence_passed,
        "Assistant → Web Search Assistant → Entry Agent completed sequentially."
        if sequence_passed
        else "The required three-agent sequential output was incomplete.",
    )

    workflow_search = core_result.get("workflow_search") or {}
    evidence_passed = (
        workflow_search.get("status") == "SUCCESS"
        and bool(workflow_search.get("items"))
    )
    add_check(
        "Workflow evidence",
        evidence_passed,
        "Current web evidence was retrieved and supplied to the Web Search Assistant."
        if evidence_passed
        else "Usable workflow search evidence was not available.",
    )

    output_validation = core_result.get("output_validation") or {}
    output_passed = bool(output_validation.get("passed"))
    add_check(
        "Output validation",
        output_passed,
        "Required outputs and final grounded answer passed structural validation."
        if output_passed
        else "One or more required outputs failed validation.",
    )

    output_guardrail = core_result.get("output_guardrail") or {}
    output_guardrail_passed = bool(output_guardrail.get("passed"))
    add_check(
        "Output guardrail",
        output_guardrail_passed,
        "Generated output passed the deterministic release gate before persistence."
        if output_guardrail_passed
        else "Generated output was blocked before persistence/evaluation.",
    )

    persistence_passed = bool(core_result.get("persistence"))
    add_check(
        "Persistence",
        persistence_passed,
        "The final grounded support record was persisted successfully."
        if persistence_passed
        else "The final support record was not persisted.",
    )

    eval_passed = bool(evaluation and evaluation.get("status") == "SUCCESS")
    add_check(
        "Output evaluation",
        eval_passed,
        "The final grounded answer was evaluated using reused workflow evidence."
        if eval_passed
        else "The Buildathon output evaluation did not complete successfully.",
    )

    temporal = core_result.get("temporal_validation") or {}
    if temporal.get("applied"):
        add_check(
            "Temporal validation",
            bool(temporal.get("passed")),
            temporal.get("detail", "Temporal validation completed."),
        )

    efficient_search = (
        core_result.get("workflow_serper_calls", 0) == 1
        and (evaluation or {}).get("serper_calls", 0) == 0
    )
    add_check(
        "API / search efficiency",
        efficient_search,
        "One workflow search was reused for generation and evaluation; no duplicate evaluation search."
        if efficient_search
        else "Search-call reuse did not match the expected efficient pattern.",
    )

    return {
        "status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
        "checks": checks,
        "llm_calls": 0,
        "search_calls": 0,
    }


# ============================================================
# EXPERIMENT 1 — DETERMINISTIC VALIDATION EVIDENCE
# ZERO ADDITIONAL MODEL / SEARCH CALLS
# ============================================================

def build_experiment1_validation(non_agentic, agentic, evaluation):
    """Validate the fixed-workflow comparison using measured run state only."""

    non_output = bool(
        (non_agentic.get("output_validation") or {}).get("passed")
    )
    agent_output = bool(
        (agentic.get("output_validation") or {}).get("passed")
    )

    non_persist = bool(non_agentic.get("persistence"))
    agent_persist = bool(agentic.get("persistence"))

    python_sequence = (
        non_agentic.get("agents") == 0
        and non_agentic.get("orchestration") == "Python"
        and bool(non_agentic.get("web_search"))
        and non_output
        and non_persist
    )

    crew_sequence = (
        agentic.get("agents") == 3
        and agentic.get("process") == "Sequential"
        and bool(agentic.get("web_search"))
        and agent_output
        and agent_persist
    )

    non_search = non_agentic.get("workflow_search") or {}
    agent_search = agentic.get("workflow_search") or {}
    shared_evidence = (
        non_search.get("search_query")
        and non_search.get("search_query") == agent_search.get("search_query")
        and non_search.get("items") == agent_search.get("items")
    )

    evaluation_passed = bool(
        evaluation
        and evaluation.get("status") == "SUCCESS"
    )

    temporal_checks = []
    for label, result in [
        ("Python", non_agentic),
        ("CrewAI", agentic),
    ]:
        temporal = result.get("temporal_validation") or {}
        if temporal.get("applied"):
            temporal_checks.append({
                "architecture": label,
                "status": temporal.get("status", "FAIL"),
                "passed": bool(temporal.get("passed")),
                "detail": temporal.get("detail", ""),
            })

    temporal_passed = all(
        check["passed"] for check in temporal_checks
    ) if temporal_checks else True

    checks = [
        {
            "name": "Python sequence",
            "status": "PASS" if python_sequence else "FAIL",
            "detail": "Generate → Search → Consolidate → Persist completed through explicit Python orchestration.",
        },
        {
            "name": "CrewAI sequence",
            "status": "PASS" if crew_sequence else "FAIL",
            "detail": "Assistant → Web Search Assistant → Entry Agent completed sequentially.",
        },
        {
            "name": "Shared workflow evidence",
            "status": "PASS" if shared_evidence else "FAIL",
            "detail": "Both architectures received the same workflow search evidence for a controlled comparison.",
        },
        {
            "name": "Output validation",
            "status": "PASS" if (non_output and agent_output) else "FAIL",
            "detail": "Required outputs were present and structurally validated without another LLM call.",
        },
        {
            "name": "Persistence",
            "status": "PASS" if (non_persist and agent_persist) else "FAIL",
            "detail": "Both workflow variants persisted their completed support record.",
        },
        {
            "name": "Common evaluation",
            "status": "PASS" if evaluation_passed else "FAIL",
            "detail": "One consolidated evaluator scored both architectures against one independent evidence set.",
        },
    ]

    if temporal_checks:
        checks.append({
            "name": "Temporal validation",
            "status": "PASS" if temporal_passed else "FAIL",
            "detail": " · ".join(
                f"{item['architecture']}: {item['detail']}"
                for item in temporal_checks
            ),
        })

    return {
        "status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
        "checks": checks,
        "temporal_checks": temporal_checks,
    }


# ============================================================
# EXPERIMENT 2 — DYNAMIC SUPPORT RESOLUTION
# Demonstrates variable tool selection and conditional escalation.
# Simulated enterprise tools are deterministic and make ZERO API calls.
# ============================================================

DYNAMIC_TOOL_TRACE = []


def _trace_tool(name: str, result: str):
    DYNAMIC_TOOL_TRACE.append({"tool": name, "result": result})
    return result


@tool("Billing Support Tool")
def billing_support_tool(issue: str) -> str:
    """Check a simulated billing system for duplicate-charge or payment issues."""
    text = issue.lower()
    if any(term in text for term in ["charged twice", "duplicate", "double charge", "twice"]):
        result = "Duplicate charge detected. Billing review/refund action is required."
    elif any(term in text for term in ["payment", "billing", "charge", "refund"]):
        result = "Billing issue detected. Payment record requires review."
    else:
        result = "No billing issue detected in the simulated billing record."
    return _trace_tool("Billing", result)


@tool("Account Access Tool")
def account_access_tool(issue: str) -> str:
    """Check a simulated account system for login, lock or access problems."""
    text = issue.lower()
    if any(term in text for term in ["locked", "can't log", "cannot log", "cant log", "login", "access"]):
        result = "Account access issue detected. Simulated account status: LOCKED."
    else:
        result = "No account-access issue detected in the simulated account record."
    return _trace_tool("Account", result)


@tool("Knowledge Support Tool")
def knowledge_support_tool(issue: str) -> str:
    """Retrieve simulated support guidance for password, setup and how-to questions."""
    text = issue.lower()
    if "password" in text:
        result = "Knowledge guidance: use the password-reset flow and verify the registered email."
    elif any(term in text for term in ["how", "setup", "configure", "help"]):
        result = "Knowledge guidance: a standard self-service support article is available."
    else:
        result = "No specific knowledge article is required for this request."
    return _trace_tool("Knowledge", result)


@tool("Human Escalation Tool")
def human_escalation_tool(issue: str) -> str:
    """Create a simulated human-review case when urgency, unresolved multi-issue risk, or manual action requires escalation."""
    result = "Human review case created: DEMO-ESC-001. No real external ticket was created."
    return _trace_tool("Escalation", result)


def run_dynamic_non_agentic(query: str):
    """Explicit Python routing: developer-written rules determine every branch."""
    start = time.perf_counter()
    text = query.lower()
    branches = []
    actions = []

    rules = [
        ("billing_rule", ["charged", "charge", "billing", "payment", "refund", "duplicate"], "Billing"),
        ("account_rule", ["locked", "login", "log in", "access", "account"], "Account"),
        ("knowledge_rule", ["password", "how", "setup", "configure", "help"], "Knowledge"),
        ("urgency_rule", ["urgent", "urgently", "immediately", "today", "client meeting", "presentation"], "Escalation"),
    ]

    tool_map = {
        "Billing": lambda: billing_support_tool.run(issue=query),
        "Account": lambda: account_access_tool.run(issue=query),
        "Knowledge": lambda: knowledge_support_tool.run(issue=query),
        "Escalation": lambda: human_escalation_tool.run(issue=query),
    }

    # Keep a separate trace because the same tools are shared with the agentic path.
    global DYNAMIC_TOOL_TRACE
    DYNAMIC_TOOL_TRACE = []

    for rule_name, keywords, tool_name in rules:
        matched = any(keyword in text for keyword in keywords)
        branches.append({"rule": rule_name, "matched": matched, "tool": tool_name})
        if matched:
            result = tool_map[tool_name]()
            actions.append({"tool": tool_name, "result": result})

    if not actions:
        # Explicit fallback encoded by developer.
        result = knowledge_support_tool.run(issue=query)
        actions.append({"tool": "Knowledge", "result": result})

    response_lines = ["Support actions completed through explicit Python routing:"]
    for action in actions:
        response_lines.append(f"- {action['tool']}: {action['result']}")

    return {
        "architecture": "Non-Agentic Dynamic",
        "answer": "\n".join(response_lines),
        "elapsed_time": time.perf_counter() - start,
        "tools_available": 4,
        "tools_used": [a["tool"] for a in actions],
        "tools_avoided": 4 - len({a["tool"] for a in actions}),
        "branches_checked": len(rules),
        "routing_rules": len(rules),
        "decision_model_calls": 0,
        "actions": actions,
        "branches": branches,
    }


def run_dynamic_agentic(query: str):
    """CrewAI agent chooses only the tools it believes are needed for the request."""
    global DYNAMIC_TOOL_TRACE
    DYNAMIC_TOOL_TRACE = []

    orchestrator = Agent(
        role="Dynamic Support Orchestrator",
        goal=(
            "Resolve the customer's support request by selecting only the necessary tools. "
            "Avoid irrelevant tools. Escalate only when urgency, unresolved multi-issue risk, "
            "or a required human action justifies it."
        ),
        backstory=(
            "You coordinate customer-support capabilities. You are given Billing, Account, "
            "Knowledge and Human Escalation tools and decide which ones are required based on "
            "the customer's actual request."
        ),
        tools=[
            billing_support_tool,
            account_access_tool,
            knowledge_support_tool,
            human_escalation_tool,
        ],
        verbose=False,
        allow_delegation=False,
        max_iter=6,
    )

    task = Task(
        description=(
            "Resolve this customer request:\n\n{query}\n\n"
            "Choose only tools that are relevant. You may use more than one tool when the request "
            "contains multiple issues. Do not call every tool by default. After observing tool results, "
            "give a concise customer-support resolution summary."
        ),
        expected_output=(
            "A concise resolution summary based on the tools actually required for the request."
        ),
        agent=orchestrator,
    )

    crew = Crew(
        agents=[orchestrator],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    start = time.perf_counter()
    crew.kickoff(inputs={"query": query})
    elapsed = time.perf_counter() - start

    trace = list(DYNAMIC_TOOL_TRACE)
    used = []
    for item in trace:
        if item["tool"] not in used:
            used.append(item["tool"])

    return {
        "architecture": "Agentic Dynamic",
        "answer": task.output.raw.strip(),
        "elapsed_time": elapsed,
        "tools_available": 4,
        "tools_used": used,
        "tools_avoided": 4 - len(used),
        "tool_trace": trace,
        "decision_model_calls": "CrewAI-controlled",
    }


def validate_agent_actions(agentic_result: dict):
    """Deterministically verify that operational claims are backed by tool execution.

    This intentionally makes no LLM/API call. It compares a small set of consequential
    action phrases in the agent's final answer with the actual CrewAI tool trace.
    """
    answer = (agentic_result.get("answer") or "").lower()
    executed = set(agentic_result.get("tools_used") or [])

    action_claims = [
        {
            "action": "Human escalation",
            "tool": "Escalation",
            "patterns": [
                r"\bi (?:have )?escalat(?:e|ed|ing)\b",
                r"\bwe (?:have )?escalat(?:e|ed|ing)\b",
                r"\bescalating this\b",
                r"\bescalated (?:this|your|the)\b",
                r"\bhuman (?:support|review) (?:case )?(?:has been |was )?created\b",
                r"\bcase (?:has been |was )?created\b",
            ],
        },
    ]

    checks = []
    for claim in action_claims:
        claimed = any(re.search(pattern, answer, flags=re.IGNORECASE) for pattern in claim["patterns"])
        tool_executed = claim["tool"] in executed
        checks.append({
            "action": claim["action"],
            "required_tool": claim["tool"],
            "claimed": claimed,
            "tool_executed": tool_executed,
            "status": "MISMATCH" if claimed and not tool_executed else "PASS",
        })

    mismatches = [item for item in checks if item["status"] == "MISMATCH"]
    return {
        "status": "MISMATCH" if mismatches else "PASS",
        "checks": checks,
        "mismatches": mismatches,
        "llm_calls": 0,
    }



def build_dynamic_decision_validation(non_agentic, agentic, action_validation):
    """Validate Experiment 2 from existing routing/tool traces only."""
    capabilities = ["Billing", "Account", "Knowledge", "Escalation"]
    capability_set = set(capabilities)

    python_tools = set(non_agentic.get("tools_used") or [])
    agent_tools = set(agentic.get("tools_used") or [])
    trace_tools = {
        item.get("tool")
        for item in (agentic.get("tool_trace") or [])
        if item.get("tool")
    }

    checks = []

    def add_check(name, passed, detail):
        checks.append({
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    python_trace_ok = (
        non_agentic.get("branches_checked") == 4
        and len(non_agentic.get("branches") or []) == 4
    )
    add_check(
        "Python rule trace",
        python_trace_ok,
        "All four developer-written routing rules were evaluated."
        if python_trace_ok
        else "The explicit Python routing trace was incomplete.",
    )

    allowed_tools_ok = agent_tools.issubset(capability_set)
    add_check(
        "Allowed capabilities",
        allowed_tools_ok,
        "The agent selected only capabilities exposed to it."
        if allowed_tools_ok
        else "The tool trace contains a capability outside the allowed set.",
    )

    trace_integrity_ok = agent_tools == trace_tools
    add_check(
        "Tool-trace integrity",
        trace_integrity_ok,
        "The summarized runtime tool list matches the actual simulated tool trace."
        if trace_integrity_ok
        else "The summarized tool list does not match the actual runtime trace.",
    )

    action_claims_ok = bool(
        action_validation
        and action_validation.get("status") == "PASS"
    )
    add_check(
        "Action-claim validation",
        action_claims_ok,
        "Consequential claims are backed by the executed tool trace."
        if action_claims_ok
        else "A consequential response claim is not proven by tool execution.",
    )

    response_ok = bool((agentic.get("answer") or "").strip())
    add_check(
        "Runtime response",
        response_ok,
        "The runtime orchestrator produced a customer-facing resolution."
        if response_ok
        else "The runtime orchestrator did not return a usable response.",
    )

    selection_rows = []
    same_count = 0

    for capability in capabilities:
        python_selected = capability in python_tools
        agent_selected = capability in agent_tools
        same = python_selected == agent_selected
        if same:
            same_count += 1

        selection_rows.append({
            "capability": capability,
            "python_selected": python_selected,
            "agent_selected": agent_selected,
            "same": same,
        })

    return {
        "status": (
            "PASS"
            if all(item["status"] == "PASS" for item in checks)
            else "FAIL"
        ),
        "checks": checks,
        "selection_rows": selection_rows,
        "selection_agreement": same_count,
        "capability_count": len(capabilities),
        "python_tools": sorted(python_tools),
        "agent_tools": sorted(agent_tools),
        "agent_tools_avoided": len(capability_set - agent_tools),
        "llm_calls": 0,
        "search_calls": 0,
    }



def run_dynamic_response_evaluation(query: str, non_agentic: dict, agentic: dict):
    """
    Experiment 2 response-quality evaluation.

    API efficiency:
    - ONE consolidated direct OpenAI evaluator call for both approaches.
    - ZERO web-search calls.
    - Grounding is measured against the actual simulated tool/rule trace,
      because Experiment 2 is about decision/tool execution rather than
      external factual retrieval.
    """
    python_trace = "\n".join(
        f"- {item['tool']}: {item['result']}"
        for item in (non_agentic.get("actions") or [])
    ) or "No simulated actions recorded."

    agent_trace = "\n".join(
        f"- {item['tool']}: {item['result']}"
        for item in (agentic.get("tool_trace") or [])
    ) or "No simulated tool calls recorded."

    prompt = f"""
You are evaluating two customer-support responses for the SAME request.

CUSTOMER REQUEST:
{query}

NON-AGENTIC / EXPLICIT PYTHON RESPONSE:
{non_agentic.get("answer", "")}

NON-AGENTIC EXECUTION TRACE:
{python_trace}

AGENTIC / RUNTIME TOOL-SELECTION RESPONSE:
{agentic.get("answer", "")}

AGENTIC EXECUTION TRACE:
{agent_trace}

This experiment uses simulated support tools. Do NOT perform web search and do
NOT judge external factual freshness. Evaluate each response only against the
customer request and its ACTUAL execution trace.

Score both approaches from 0 to 100 on:

RELEVANCE
- Does the response address the customer's actual request?

COMPLETENESS
- Does it cover the important issues in the request that its execution trace
  shows were handled?

CONSISTENCY
- Is the response internally coherent and non-contradictory?

TRACE_GROUNDING
- Are operational claims in the response supported by the ACTUAL simulated
  tool/rule trace?
- Penalize claims of actions that were not executed.
- Do not penalize merely because no web evidence exists; web grounding is not
  the purpose of this experiment.

Return ONLY valid JSON in exactly this shape:
{{
  "non_agentic": {{
    "relevance": {{"score": 0, "reason": ""}},
    "completeness": {{"score": 0, "reason": ""}},
    "consistency": {{"score": 0, "reason": ""}},
    "groundedness": {{"score": 0, "reason": ""}}
  }},
  "agentic": {{
    "relevance": {{"score": 0, "reason": ""}},
    "completeness": {{"score": 0, "reason": ""}},
    "consistency": {{"score": 0, "reason": ""}},
    "groundedness": {{"score": 0, "reason": ""}}
  }},
  "comparison": {{
    "available": true,
    "agreement": "AGREE",
    "reason": ""
  }}
}}

Allowed comparison agreement values:
AGREE, PARTIAL, DISAGREE

All scores must be integers from 0 to 100.
Keep reasons concise and trace-focused.
"""

    start = time.perf_counter()

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        result = parse_json_response(
            response.choices[0].message.content
        )

        for architecture in ["non_agentic", "agentic"]:
            architecture_result = result.get(architecture) or {}

            for metric in [
                "relevance",
                "completeness",
                "consistency",
                "groundedness",
            ]:
                metric_result = architecture_result.get(metric) or {}
                architecture_result[metric] = {
                    "score": clamp_score(metric_result.get("score", 0)),
                    "reason": str(metric_result.get("reason", "")).strip(),
                }

            result[architecture] = architecture_result

        comparison = result.get("comparison") or {}
        agreement = str(
            comparison.get("agreement", "PARTIAL")
        ).strip().upper()

        if agreement not in {"AGREE", "PARTIAL", "DISAGREE"}:
            agreement = "PARTIAL"

        result["comparison"] = {
            "available": True,
            "agreement": agreement,
            "reason": str(comparison.get("reason", "")).strip(),
        }

        return {
            "status": "SUCCESS",
            "result": result,
            "elapsed_time": time.perf_counter() - start,
            "openai_calls": 1,
            "serper_calls": 0,
            "evaluation_mode": "tool_trace",
            "grounding_label": "Trace Grounding",
        }

    except Exception as error:
        return {
            "status": "ERROR",
            "result": {},
            "elapsed_time": time.perf_counter() - start,
            "openai_calls": 1,
            "serper_calls": 0,
            "evaluation_mode": "tool_trace",
            "grounding_label": "Trace Grounding",
            "error": str(error),
        }


def build_dynamic_observations(non_agentic, agentic, action_validation=None):
    observations = []
    observations.append({
        "title": "Decision ownership",
        "text": (
            f"The non-agentic path evaluated {non_agentic['branches_checked']} developer-written routing rules. "
            "The agentic path received the available capabilities and selected tools at runtime."
        ),
    })
    observations.append({
        "title": "Tool selection",
        "text": (
            f"Python routing used: {', '.join(non_agentic['tools_used']) or 'None'}. "
            f"Agent-selected tools: {', '.join(agentic['tools_used']) or 'None'}."
        ),
    })
    observations.append({
        "title": "Latency",
        "text": (
            f"Explicit routing completed in {non_agentic['elapsed_time']:.2f}s; "
            f"agentic decision/orchestration completed in {agentic['elapsed_time']:.2f}s."
        ),
    })
    observations.append({
        "title": "Learning takeaway",
        "text": (
            "Both approaches can implement conditional workflows. The difference is where decision logic lives: "
            "explicit Python rules versus runtime agent/tool selection. As tools and combinations grow, this changes "
            "the maintainability, predictability, latency and governance trade-offs."
        ),
    })
    if action_validation:
        if action_validation["status"] == "PASS":
            validation_text = (
                "PASS — no unsupported consequential action claim was detected. "
                "The check is deterministic and uses the actual tool-execution trace; it makes zero LLM calls."
            )
        else:
            missing = ", ".join(
                f"{item['action']} requires {item['required_tool']}"
                for item in action_validation["mismatches"]
            )
            validation_text = (
                f"MISMATCH — the final agent response claims an operational action not proven by the tool trace: {missing}. "
                "This check is deterministic and makes zero LLM calls."
            )
        observations.append({"title": "Action validation", "text": validation_text})
    return observations




def render_experiment_overview(story_type: str):
    """Render a compact zero-API overview band that explains the selected experiment."""
    if story_type == "fixed":
        experiment_label = "EXPERIMENT 1"
        experiment_title = "Fixed Sequential Workflow"
        cards = [
            (
                "🎯",
                "What is being tested",
                "The same predictable support task follows a known sequence every time.",
                "Known path",
            ),
            (
                "🐍",
                "Non-Agentic approach",
                "Python explicitly orchestrates generation, search, consolidation and persistence.",
                "Developer-owned flow",
            ),
            (
                "🤖",
                "Agentic approach",
                "CrewAI executes the same known sequence through specialized role-based hand-offs.",
                "Agent hand-offs",
            ),
        ]
        question = "Does a fixed, predictable workflow really need agents?"
        focus = "Compare quality, latency and orchestration overhead when the execution path is already known."
    else:
        experiment_label = "EXPERIMENT 2"
        experiment_title = "Dynamic Decision Workflow"
        cards = [
            (
                "🧭",
                "What is being tested",
                "The execution path changes with each request and may require different capabilities.",
                "Variable path",
            ),
            (
                "📋",
                "Non-Agentic approach",
                "Python evaluates developer-written routing rules and invokes matching capabilities.",
                "Rules at design time",
            ),
            (
                "🧠",
                "Agentic approach",
                "The agent interprets the request and selects useful tools and actions at runtime.",
                "Runtime selection",
            ),
        ]
        question = "When does runtime decision-making make agents more useful?"
        focus = "Compare explicit routing rules with runtime reasoning as capability combinations grow."

    card_html = "".join(
        f"""
        <div class='exp-overview-card'>
          <div class='exp-overview-icon'>{icon}</div>
          <div class='exp-overview-copy'>
            <div class='exp-overview-heading'>{heading}</div>
            <div class='exp-overview-text'>{body}</div>
            <span class='exp-overview-tag'>{tag}</span>
          </div>
        </div>
        """
        for icon, heading, body, tag in cards
    )

    st.markdown(
        f"""
        <div class='exp-overview-shell'>
          <div class='exp-overview-top'>
            <div>
              <div class='exp-overview-kicker'>{experiment_label}</div>
              <div class='exp-overview-title'>{experiment_title}</div>
            </div>
            <div class='exp-overview-focus'>{focus}</div>
          </div>
          <div class='exp-overview-grid'>{card_html}</div>
          <div class='exp-learning-question'><span>Learning question</span>{question}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_experiment_architecture(experiment_type: str):
    """
    Render the experiment architecture as an optional collapsed reference.

    The architecture images are local UI assets:
    - zero OpenAI calls
    - zero search calls
    - collapsed by default to keep experiment pages compact
    """
    architecture_path = EXPERIMENT_ARCHITECTURES[experiment_type]

    st.markdown("### Architecture")
    st.caption(
        "Optional visual reference · rendered locally · "
        "zero additional OpenAI or search calls"
    )

    with st.expander(
        "View Architecture Comparison",
        expanded=False,
    ):
        st.markdown(
            """
            <div class="architecture-callout">
                <div class="architecture-callout-icon">⌘</div>
                <div>
                    <div class="architecture-callout-title">
                        Architecture comparison
                    </div>
                    <div class="architecture-callout-text">
                        Same business request, clearly separated orchestration
                        models and decision ownership.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if architecture_path.exists():
            # Keep the image compact even when the expander is opened.
            left, center, right = st.columns([1.25, 2.50, 1.25])

            with center:
                st.image(
                    str(architecture_path),
                    use_container_width=True,
                )
        else:
            st.warning(
                f"Architecture image not found: {architecture_path}. "
                "Keep the assets folder beside app.py."
            )


def render_scenario_story(story_type: str):
    """Render a zero-API pictorial story explaining the workflow."""
    stories = {
        "buildathon": {
            "title": "Scenario Story · How the support request moves through the crew",
            "customer": "What is the latest stable version of Python?",
            "steps": [
                ("🤖", "Assistant Agent", "Creates the first answer from model knowledge."),
                ("🌐", "Web Search Assistant", "Checks current web information and enriches the answer."),
                ("📝", "Entry Agent", "Synthesizes one grounded final answer and persists the support record."),
            ],
            "outcome": "Customer receives the support result · record saved to answers.txt",
        },
        "fixed": {
            "title": "Scenario Story · Same predictable task, two orchestration styles",
            "customer": "What is the latest stable version of Python?",
            "left_title": "Non-Agentic · Explicit Python",
            "left_steps": "🐍 Generate → 🌐 Search → 🧩 Consolidate → 💾 Persist",
            "left_note": "The developer already knows the sequence and encodes it directly.",
            "right_title": "Agentic · CrewAI",
            "right_steps": "🤖 Assistant → 🌐 Search Agent → 📝 Entry Agent",
            "right_note": "Agents perform the same known sequence through role-based hand-offs.",
            "outcome": "Both can solve the predictable workflow; the experiment measures quality, latency and orchestration overhead.",
        },
        "dynamic": {
            "title": "Scenario Story · One customer request can require different capabilities",
            "customer": "I was charged twice, I can't log in, and I need access urgently for a customer presentation.",
            "left_title": "Non-Agentic · Developer-Written Routing",
            "left_steps": "📋 Billing rule → 📋 Account rule → 📋 Knowledge rule → 📋 Urgency rule",
            "left_note": "Python evaluates rules written in advance and invokes the matching capabilities.",
            "right_title": "Agentic · Runtime Selection",
            "right_steps": "🧠 Interpret request → 💳 Billing → 🔐 Account → 🚨 Escalation · skip irrelevant tools",
            "right_note": "The agent receives available capabilities and chooses the useful combination at runtime.",
            "outcome": "The difference is decision ownership: predefined routing rules versus runtime reasoning and tool selection.",
        },
    }
    data = stories[story_type]
    st.markdown(f"### {data['title']}")
    st.caption("Pictorial learning view · rendered locally in the UI · zero additional OpenAI or search calls")

    if story_type == "buildathon":
        cards = "".join(
            f"<div class='story-arrow'>→</div><div class='story-card'><div class='story-icon'>{icon}</div><b>{name}</b><span>{desc}</span></div>"
            for icon, name, desc in data["steps"]
        )
        html = f"""
        <div class='story-shell'>
          <div class='story-customer'><div class='story-person'>👨‍💼</div><div><b>Customer</b><div class='story-bubble'>“{data['customer']}”</div></div></div>
          <div class='story-arrow'>→</div>
          {cards}
        </div>
        <div class='story-outcome'>✅ {data['outcome']}</div>
        """
    else:
        html = f"""
        <div class='story-customer story-customer-wide'><div class='story-person'>👨‍💼</div><div><b>Customer</b><div class='story-bubble'>“{data['customer']}”</div></div></div>
        <div class='story-down'>↓</div>
        <div class='story-split'>
          <div class='story-lane'>
            <div class='story-lane-title'>{data['left_title']}</div>
            <div class='story-route'>{data['left_steps']}</div>
            <div class='story-note'>{data['left_note']}</div>
          </div>
          <div class='story-vs'>VS</div>
          <div class='story-lane'>
            <div class='story-lane-title'>{data['right_title']}</div>
            <div class='story-route'>{data['right_steps']}</div>
            <div class='story-note'>{data['right_note']}</div>
          </div>
        </div>
        <div class='story-down'>↓</div>
        <div class='story-outcome'>💡 {data['outcome']}</div>
        """
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# ENTERPRISE LEARNING LAB UI
# ============================================================

st.markdown(
    """
    <style>
    /* =========================================================
       GLOBAL TYPOGRAPHY SYSTEM
       One professional system font stack across the full app.
       ========================================================= */
    :root{
        --app-font:-apple-system,BlinkMacSystemFont,"Segoe UI","Helvetica Neue",Arial,sans-serif;
        --text-primary:#17223F;
        --text-secondary:#53617A;
    }

    html,
    body,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] *,
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] *,
    .stApp,
    .stApp *{
        font-family:var(--app-font)!important;
    }

    /* Keep Streamlit Material icons as icons, not visible text names
       such as "arrow_right". */
    [data-testid="stIconMaterial"],
    [data-testid="stIconMaterial"] *,
    span[data-testid="stIconMaterial"]{
        font-family:"Material Symbols Rounded","Material Symbols Outlined"!important;
        font-weight:normal!important;
        font-style:normal!important;
        font-size:inherit;
        line-height:1!important;
        letter-spacing:normal!important;
        text-transform:none!important;
        white-space:nowrap!important;
        word-wrap:normal!important;
        direction:ltr!important;
        font-feature-settings:"liga"!important;
        -webkit-font-feature-settings:"liga"!important;
        -webkit-font-smoothing:antialiased!important;
    }

    /* Keep code-like content intentionally monospace. */
    code,
    pre,
    kbd,
    samp,
    [data-testid="stCode"] *{
        font-family:"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace!important;
    }

    /* Main page headings */
    .stApp h1{
        font-family:var(--app-font)!important;
        font-size:34px!important;
        line-height:1.15!important;
        font-weight:800!important;
        letter-spacing:-.02em!important;
        color:#101D3A!important;
    }

    .stApp h2{
        font-family:var(--app-font)!important;
        font-size:26px!important;
        line-height:1.2!important;
        font-weight:800!important;
        letter-spacing:-.015em!important;
        color:#101D3A!important;
    }

    .stApp h3{
        font-family:var(--app-font)!important;
        font-size:21px!important;
        line-height:1.25!important;
        font-weight:750!important;
        letter-spacing:-.01em!important;
        color:#14213D!important;
    }

    .stApp h4{
        font-family:var(--app-font)!important;
        font-size:16px!important;
        line-height:1.3!important;
        font-weight:750!important;
        color:#17223F!important;
    }

    /* Standard body copy */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stText"],
    .stApp label,
    .stApp input,
    .stApp textarea,
    .stApp select{
        font-family:var(--app-font)!important;
        font-size:14px;
        line-height:1.55;
        color:var(--text-secondary);
    }

    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] b{
        font-family:var(--app-font)!important;
        font-weight:700!important;
        color:var(--text-primary);
    }

    /* Streamlit widget labels */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stTextArea"] label p,
    [data-testid="stTextInput"] label p,
    [data-testid="stSelectbox"] label p,
    [data-testid="stRadio"] > label p{
        font-family:var(--app-font)!important;
        font-size:14px!important;
        font-weight:650!important;
        color:#263853!important;
    }

    /* Expander titles */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] summary p{
        font-family:var(--app-font)!important;
        font-size:14px!important;
        font-weight:650!important;
        line-height:1.35!important;
        color:#334763!important;
    }

    /* Alerts / status messages */
    [data-testid="stAlert"],
    [data-testid="stAlert"] *,
    [data-testid="stAlert"] p{
        font-family:var(--app-font)!important;
        font-size:13px!important;
        line-height:1.5!important;
    }

    /* Buttons */
    .stButton button,
    .stFormSubmitButton button,
    .stButton button *,
    .stFormSubmitButton button *{
        font-family:var(--app-font)!important;
    }

    /* Metrics */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] *{
        font-family:var(--app-font)!important;
    }

    /* ===== Exact reference layout ===== */
    :root{--ink:#101d3a;--muted:#586985;--line:#d8e1ed;--blue:#1B60FB;--soft:#f7faff;}
    html,body,[class*="css"]{font-size:16px;}
    [data-testid="stAppViewContainer"]{background:#fff;}
    [data-testid="stHeader"]{background:transparent;height:0;}
    [data-testid="stToolbar"]{display:none;}
    [data-testid="stMain"]{margin-left:0!important;}
    [data-testid="stMainBlockContainer"],.block-container{
        max-width:none!important;
        width:100%!important;
        padding:1.1rem 1.5rem 3.2rem 1.5rem!important;
        margin:0!important;
    }
    h1,h2,h3,h4{letter-spacing:-.025em;color:var(--ink)!important;}
    h1{font-size:36px!important;line-height:1.08!important;}
    h2{font-size:28px!important;line-height:1.15!important;}
    h3{font-size:23px!important;line-height:1.18!important;}
    h4{font-size:18px!important;}
    [data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,
    [data-testid="stText"],label{font-size:15px;line-height:1.55;color:#42516d;}
    [data-testid="stCaptionContainer"]{font-size:14px!important;line-height:1.48!important;color:#6a7891!important;}

    /* ===== Sidebar — matched to supplied reference ===== */
    [data-testid="stSidebar"]{
        background:#F4F7FC!important;
        border-right:1px solid #D8E0EC!important;
        min-width:258px!important;max-width:258px!important;width:258px!important;
    }
    [data-testid="stSidebar"]>div:first-child{width:258px!important;padding-top:.8rem!important;}
    [data-testid="stSidebar"] .block-container{padding:.9rem 1rem 1.45rem!important;}

    .side-brand{display:flex;gap:10px;align-items:center;margin:0 0 12px;padding:3px 2px 14px;border-bottom:1px solid #D5DEEA}
    .side-logo{width:33px;height:33px;color:#17325E;display:flex;align-items:center;justify-content:center}
    .side-logo svg{width:31px;height:31px;fill:none;stroke:currentColor;stroke-width:2.8;stroke-linecap:round;stroke-linejoin:round}
    .side-title{font-size:18px;font-weight:800;line-height:1.08;color:#101C3A}.side-sub{font-size:11px;color:#7A879D;margin-top:4px}

    /* Custom HTML navigation gives exact icon + subtitle layout */
    .ref-nav{display:flex;flex-direction:column;gap:5px;margin:0 0 18px 0}
    .ref-nav-item{display:grid;grid-template-columns:31px 1fr;column-gap:9px;align-items:center;text-decoration:none!important;padding:8px 8px;border-radius:9px;min-height:58px;box-sizing:border-box;color:#111F40!important}
    .ref-nav-item:hover{background:#EEF2F8;text-decoration:none!important}
    .ref-nav-item.active{background:#FDE6E9!important}
    .ref-nav-icon{width:27px;height:27px;display:flex;align-items:center;justify-content:center}
    .ref-nav-icon svg{width:27px;height:27px;display:block}
    .ref-nav-title{font-size:16px;font-weight:800;line-height:1.12;color:#101D3A;margin:0}
    .ref-nav-item.active .ref-nav-title{color:#C82438}
    .ref-nav-sub{font-size:11px;line-height:1.25;color:#42516D;margin-top:3px;font-weight:550}

    .side-kicker{font-size:12px;font-weight:800;letter-spacing:.12em;color:#4A5872;margin:22px 5px 12px;text-transform:uppercase}
    .side-copy{font-size:15px;line-height:1.45;color:#66748D;padding:0 5px 19px;border-bottom:1px solid #D5DEEA}

    .tech-stack{padding:0 4px 18px;border-bottom:1px solid #D5DEEA}
    .tech-row{display:grid;grid-template-columns:28px 1fr;align-items:center;gap:7px;min-height:30px;color:#17223F;font-size:15px;font-weight:600;line-height:1.25}
    .tech-icon{width:24px;height:24px;display:flex;align-items:center;justify-content:center}
    .tech-icon svg{width:23px;height:23px;display:block}
    .side-quote{display:none!important}
    .guardrail-panel{margin:0 4px 18px;padding:12px;border:1px solid #D6E2F5;border-radius:11px;background:#F8FBFF}
    .guardrail-head{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:9px}
    .guardrail-title{font-size:13px;font-weight:800;letter-spacing:.07em;text-transform:uppercase;color:#405577}
    .guardrail-badge{font-size:11px;font-weight:800;padding:4px 8px;border-radius:999px}
    .guardrail-badge.pass{background:#E8F7EE;color:#257A46}.guardrail-badge.block{background:#FDEBEC;color:#B23A48}.guardrail-badge.pending{background:#EEF3FB;color:#6A7891}
    .guardrail-row{display:grid;grid-template-columns:17px 1fr;gap:6px;align-items:start;padding:4px 0;font-size:12px;line-height:1.3;color:#53617A}
    .guardrail-row .ok{color:#2F9E5B;font-weight:800}.guardrail-row .bad{color:#D64555;font-weight:800}
    .guardrail-ai{margin-top:8px;padding-top:8px;border-top:1px solid #DCE6F3;font-size:12px;color:#53617A}.guardrail-ai strong{color:#17223F}
    .eval-side-panel{margin:0 4px 18px;padding:12px;border:1px solid #D6E2F5;border-radius:11px;background:#F8FBFF}
    .eval-side-head{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px}
    .eval-side-title{font-size:13px;font-weight:800;letter-spacing:.07em;text-transform:uppercase;color:#405577}
    .eval-side-badge{font-size:11px;font-weight:800;padding:4px 8px;border-radius:999px;background:#EAF0FF;color:#4F63D8}
    .eval-side-row{display:flex;align-items:flex-start;justify-content:space-between;gap:8px;padding:4px 0;font-size:12px;line-height:1.3;color:#53617A}
    .eval-side-row span{max-width:145px}
    .eval-side-row strong{color:#17223F}
    .eval-side-note{margin-top:8px;padding-top:8px;border-top:1px solid #DCE6F3;font-size:11px;line-height:1.35;color:#73809A}
    .eval-evidence{margin:12px 0 4px;padding:12px 14px;border:1px solid #D7E4FF;border-left:4px solid #5D73F7;border-radius:9px;background:linear-gradient(90deg,#EEF4FF,#F8FAFF);font-size:14px;line-height:1.45;color:#53617A}

    /* ===== Header / hero ===== */
    .enterprise-hero{
        border:1px solid #dce3ec;border-radius:0 0 14px 14px;
        padding:14px 26px 20px;background:linear-gradient(135deg,#fff 0%,#f5f8fc 100%);
        margin:-1.1rem -1.5rem 0 -1.5rem;box-shadow:0 3px 14px rgba(30,50,80,.04)
    }
    .hero-grid{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:34px;align-items:center}
    .eyebrow{font-size:11px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:#586984;margin-bottom:10px}
    .hero-title{font-size:36px;font-weight:800;color:#101d3a;line-height:1.08;margin-bottom:11px}
    .hero-copy{font-size:15px;line-height:1.55;color:#53617a;max-width:900px}
    .tag{display:inline-block;padding:6px 13px;margin:12px 7px 0 0;border-radius:999px;border:1px solid #d5deea;font-size:13px;background:#fff;color:#273b5b}
    .hero-quote{background:#EDF4FF;border-radius:13px;padding:18px 20px;color:#244FD8;font-size:15px;font-style:italic;line-height:1.55}
    .hero-quote small{display:block;text-align:right;color:#67758d;font-style:normal;margin-top:8px;font-size:13px}

    /* ===== Section headings ===== */
    .section-title{font-size:28px;font-weight:800;color:#101d3a;margin:10px 0 5px}
    .section-sub{font-size:15px;line-height:1.5;color:#61708a;margin-bottom:14px}

    /* ===== Experiment architecture image section ===== */
    .architecture-subtitle{
        margin:-4px 0 12px;
        font-size:14px;
        line-height:1.5;
        font-weight:500;
        color:#8A97AC;
    }
    .architecture-callout{
        display:flex;
        align-items:center;
        gap:12px;
        max-width:760px;
        box-sizing:border-box;
        margin:0 0 16px;
        padding:11px 14px;
        border:1px solid #D7E4FF;
        border-left:4px solid #5D73F7;
        border-radius:9px;
        background:linear-gradient(90deg,#EEF4FF 0%,#F7F9FF 100%);
        box-shadow:0 1px 3px rgba(42,74,160,.04);
    }
    .architecture-callout-icon{
        flex:0 0 34px;
        width:34px;
        height:34px;
        border-radius:9px;
        display:flex;
        align-items:center;
        justify-content:center;
        background:#E3EBFF;
        color:#3C5BF8;
        font-size:17px;
        font-weight:800;
    }
    .architecture-callout-title{
        margin:0 0 2px;
        font-size:12px;
        line-height:1.2;
        font-weight:800;
        letter-spacing:.08em;
        text-transform:uppercase;
        color:#405577;
    }
    .architecture-callout-text{
        margin:0;
        font-size:14px;
        line-height:1.45;
        font-weight:550;
        color:#53617A;
    }

    /* ===== Buildathon architecture ===== */
    .arch-shell{border:1px solid #dce3ec;border-radius:13px;padding:16px;background:linear-gradient(180deg,#fbfdff,#f7faff);margin:10px 0 20px;overflow:hidden}
    .arch-flow{display:grid;grid-template-columns:minmax(112px,1fr) 24px minmax(122px,1.04fr) 24px minmax(126px,1.06fr) 24px minmax(126px,1.06fr) 24px minmax(126px,1.06fr) 24px minmax(126px,1.06fr) 24px minmax(112px,1fr);gap:5px;align-items:center}
    .arch-card{border:1px solid #d5dfeb;border-radius:11px;padding:16px 12px;background:#fff;text-align:center;min-height:195px;display:flex;flex-direction:column;justify-content:center;box-sizing:border-box}
    .arch-card.blue{background:#EEF5FF;border-color:#9FC1FF;box-shadow:inset 0 0 0 1px rgba(27,96,251,.05)}.arch-card.green{background:#EFF9ED;border-color:#B7DBB4}.arch-card.purple{background:#F3F0FF;border-color:#C6B7FF}.arch-card.output-guard{background:#FFFFFF;border-color:#D9E1EC;box-shadow:none}
    .arch-icon{font-size:38px;margin-bottom:10px}.arch-icon.blue-icon{color:#1B60FB}.arch-icon.guard-icon{color:#42567A}.arch-icon.green-icon{color:#008C58}.arch-icon.purple-icon{color:#5A43E8}.arch-icon.output-guard-icon{color:#4F5FF5}.arch-icon svg{width:42px;height:42px;display:block;margin:0 auto;stroke:currentColor;fill:none;stroke-width:2.2;stroke-linecap:round;stroke-linejoin:round}.arch-icon svg.fill-icon{fill:currentColor;stroke:none}.arch-name{font-size:17px;font-weight:800;color:#101d3a;line-height:1.2}.arch-card.blue .arch-name{color:#173EAE}.arch-card.green .arch-name{color:#101d3a}.arch-card.purple .arch-name{color:#101d3a}.arch-card.output-guard .arch-name{color:#101d3a}.arch-stage{font-size:15px;font-weight:800;line-height:1.2;margin-top:3px}.arch-stage.blue-stage{color:#1B60FB}.arch-stage.green-stage{color:#008C58}.arch-stage.purple-stage{color:#5A43E8}.arch-stage.output-guard-stage{color:#4F5FF5}.arch-role{font-size:14px;color:#576983;line-height:1.48;margin-top:7px}.arch-badge{display:inline-block;margin:12px auto 0;padding:5px 11px;border:1px solid #cdd8e6;border-radius:999px;font-size:12px;color:#2e4f80;background:rgba(255,255,255,.82)}
    .arch-arrow{text-align:center;font-size:30px;color:#98a5b7;font-weight:800}

    /* ===== Metrics ===== */
    [data-testid="stMetric"]{border:1px solid #dce3ec;border-radius:11px;padding:16px 17px!important;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.02);min-height:104px}
    [data-testid="stMetricLabel"] p{font-size:14px!important;font-weight:700!important;color:#263a5a!important;}
    [data-testid="stMetricValue"]{font-size:30px!important;font-weight:800!important;color:#0f1d3a!important;}
    .callout{background:#EDF4FF;border:1px solid #D4E2FF;border-radius:9px;padding:13px 15px;color:#244FD8;font-size:14px;margin:12px 0 20px}

    /* ===== Inputs / buttons ===== */
    [data-testid="stTextArea"] textarea{font-size:15px!important;line-height:1.5!important;min-height:105px!important;background:#f4f7fb!important;border:0!important;border-radius:8px!important;padding:14px!important;}
    [data-testid="stTextArea"] label p{font-size:15px!important;font-weight:600!important;color:#263853!important;}
    /* Compact workflow buttons — approved blue theme.
       Covers current and older Streamlit button DOM structures. */
    div.stButton,
    div.stFormSubmitButton,
    [data-testid="stButton"],
    [data-testid="stFormSubmitButton"]{
        width:fit-content!important;
    }

    div.stButton>button,
    div.stFormSubmitButton>button,
    [data-testid="stButton"] button,
    [data-testid="stFormSubmitButton"] button,
    button[data-testid="stBaseButton-primary"]{
        background:#4F5FF5!important;
        border:1px solid #4F5FF5!important;
        border-radius:8px!important;
        min-height:38px!important;
        height:38px!important;
        width:auto!important;
        min-width:178px!important;
        padding:0 18px!important;
        font-family:var(--app-font)!important;
        font-weight:750!important;
        color:#FFFFFF!important;
        font-size:14px!important;
        line-height:1!important;
        box-shadow:0 2px 6px rgba(79,95,245,.20)!important;
    }

    div.stButton>button:hover,
    div.stFormSubmitButton>button:hover,
    [data-testid="stButton"] button:hover,
    [data-testid="stFormSubmitButton"] button:hover,
    button[data-testid="stBaseButton-primary"]:hover{
        background:#4050DC!important;
        border-color:#4050DC!important;
        color:#FFFFFF!important;
        box-shadow:0 3px 9px rgba(79,95,245,.26)!important;
    }

    div.stButton>button:active,
    div.stFormSubmitButton>button:active,
    [data-testid="stButton"] button:active,
    [data-testid="stFormSubmitButton"] button:active,
    button[data-testid="stBaseButton-primary"]:active{
        background:#3443C7!important;
        border-color:#3443C7!important;
        color:#FFFFFF!important;
    }

    div.stButton>button *,
    div.stFormSubmitButton>button *,
    [data-testid="stButton"] button *,
    [data-testid="stFormSubmitButton"] button *,
    button[data-testid="stBaseButton-primary"] *{
        color:#FFFFFF!important;
        fill:#FFFFFF!important;
        font-family:var(--app-font)!important;
        font-weight:750!important;
    }


    /* ===== Experiment selector — two immediately visible visual cards ===== */
    div[data-testid="stRadio"]:has(div[role="radiogroup"]) > label{
        font-size:15px!important;
        font-weight:750!important;
        color:#263853!important;
        margin-bottom:10px!important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"]{
        display:grid!important;
        grid-template-columns:repeat(2,minmax(0,1fr))!important;
        gap:16px!important;
        align-items:stretch!important;
        width:100%!important;
        margin:2px 0 18px!important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label{
        position:relative!important;
        display:flex!important;
        align-items:flex-start!important;
        width:100%!important;
        min-height:128px!important;
        box-sizing:border-box!important;
        padding:20px 20px 48px 20px!important;
        border:1px solid #D7DFEA!important;
        border-radius:14px!important;
        background:#FFFFFF!important;
        box-shadow:0 3px 12px rgba(22,40,72,.045)!important;
        transition:border-color .18s ease, box-shadow .18s ease, transform .18s ease, background .18s ease!important;
        cursor:pointer!important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover{
        border-color:#AAB9D2!important;
        box-shadow:0 6px 18px rgba(22,40,72,.08)!important;
        transform:translateY(-1px)!important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked){
        border:2px solid #5D73F7!important;
        background:linear-gradient(135deg,#F7F9FF 0%,#EEF2FF 100%)!important;
        box-shadow:0 7px 20px rgba(60,91,248,.13)!important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label p{
        font-size:17px!important;
        line-height:1.28!important;
        font-weight:750!important;
        color:#13213F!important;
        margin:0!important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p{
        color:#2D49D7!important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(1)::after,
    div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(2)::after{
        position:absolute!important;
        left:52px!important;
        right:18px!important;
        bottom:16px!important;
        font-size:13px!important;
        line-height:1.38!important;
        font-weight:550!important;
        color:#66758E!important;
        letter-spacing:0!important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(1)::after{
        content:"Predictable path • fixed sequence • compare explicit Python orchestration with CrewAI";
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(2)::after{
        content:"Dynamic path • runtime decisions • test agent-led capability and tool selection";
    }
    @media(max-width:900px){
        div[data-testid="stRadio"] div[role="radiogroup"]{
            grid-template-columns:1fr!important;
        }
    }

    /* ===== Experiment overview band ===== */
    .exp-overview-shell{
        border:1px solid #DCE3EC;
        border-radius:14px;
        padding:18px 20px 16px;
        margin:8px 0 16px;
        background:linear-gradient(135deg,#FBFDFF 0%,#F5F8FF 100%);
        box-shadow:0 3px 14px rgba(22,40,72,.04);
    }
    .exp-overview-top{
        display:grid;
        grid-template-columns:minmax(0,1fr) minmax(280px,.9fr);
        gap:22px;
        align-items:center;
        margin-bottom:14px;
    }
    .exp-overview-kicker{
        font-size:11px;
        font-weight:800;
        letter-spacing:.13em;
        color:#5B6E8B;
        margin-bottom:4px;
    }
    .exp-overview-title{
        font-size:22px;
        line-height:1.2;
        font-weight:800;
        color:#101D3A;
    }
    .exp-overview-focus{
        border-left:3px solid #5D73F7;
        padding:8px 0 8px 13px;
        font-size:14px;
        line-height:1.5;
        color:#53617A;
    }
    .exp-overview-grid{
        display:grid;
        grid-template-columns:repeat(3,minmax(0,1fr));
        gap:12px;
    }
    .exp-overview-card{
        display:grid;
        grid-template-columns:42px 1fr;
        gap:11px;
        align-items:start;
        min-height:116px;
        padding:15px;
        border:1px solid #D9E2EF;
        border-radius:11px;
        background:#FFFFFF;
        box-sizing:border-box;
    }
    .exp-overview-icon{
        width:38px;
        height:38px;
        border-radius:10px;
        display:flex;
        align-items:center;
        justify-content:center;
        background:#EEF3FF;
        font-size:21px;
    }
    .exp-overview-heading{
        font-size:15px;
        font-weight:750;
        line-height:1.25;
        color:#17223F;
        margin-bottom:5px;
    }
    .exp-overview-text{
        font-size:13px;
        line-height:1.45;
        color:#66758E;
        min-height:56px;
    }
    .exp-overview-tag{
        display:inline-block;
        margin-top:8px;
        padding:4px 8px;
        border-radius:999px;
        border:1px solid #D7E1F0;
        background:#F7F9FC;
        color:#405577;
        font-size:11px;
        font-weight:750;
    }
    .exp-learning-question{
        margin-top:13px;
        border-radius:9px;
        background:#EAF1FF;
        color:#274FC8;
        padding:10px 13px;
        font-size:14px;
        line-height:1.4;
        font-weight:650;
    }
    .exp-learning-question span{
        display:inline-block;
        margin-right:9px;
        font-size:11px;
        letter-spacing:.08em;
        text-transform:uppercase;
        font-weight:800;
        color:#5D73F7;
    }
    @media(max-width:1000px){
        .exp-overview-top{grid-template-columns:1fr;gap:10px}
        .exp-overview-grid{grid-template-columns:1fr}
        .exp-overview-text{min-height:0}
    }

    /* ===== Story / experiment visuals ===== */
    .story-shell{display:flex;align-items:center;gap:14px;margin:10px 0 10px}.story-customer{display:flex;align-items:center;gap:14px;border:1px solid #dce3ec;border-radius:12px;padding:18px;background:#fff;min-width:265px}.story-customer-wide{max-width:980px;margin:10px auto 0}.story-person{font-size:46px}.story-bubble{margin-top:7px;padding:11px 14px;border-radius:10px;background:#f4f7fb;font-size:15px;line-height:1.45}.story-card{flex:1;border:1px solid #dce3ec;border-radius:12px;padding:18px;background:#fff;min-height:142px}.story-card span{display:block;margin-top:8px;font-size:14px;line-height:1.5;color:#61708a}.story-icon{font-size:34px;margin-bottom:8px}.story-arrow,.story-down{font-size:28px;text-align:center;color:#9ca8b9}.story-split{display:grid;grid-template-columns:1fr 62px 1fr;gap:16px;align-items:stretch}.story-lane{border:1px solid #dce3ec;border-radius:12px;padding:20px;background:#fff}.story-lane-title{font-size:20px;font-weight:800;margin-bottom:13px;color:#15213d}.story-route{padding:15px;border-radius:10px;background:#f7f9fc;line-height:1.8;font-size:15px}.story-note{margin-top:13px;font-size:14px;line-height:1.55;color:#66748a}.story-vs{display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:800;color:#96a2b3}.story-outcome{border:1px solid #9FC1FF;background:#EEF5FF;border-radius:10px;padding:14px 16px;margin:10px 0 18px;font-size:15px;color:#244FD8}

    /* ===== General panels ===== */
    .panel{border:1px solid #dce3ec;border-radius:12px;padding:20px;background:#fff;margin:12px 0}
    [data-testid="stExpander"] summary p{font-size:15px!important;}

    /* ===== Expander layout fix =====
       Prevent arrow/icon + title text overlap across all expanders. */
    [data-testid="stExpander"] details > summary{
        display:flex!important;
        align-items:center!important;
        gap:10px!important;
        min-height:42px!important;
        padding:8px 12px!important;
        box-sizing:border-box!important;
        overflow:visible!important;
    }

    [data-testid="stExpander"] details > summary > div{
        display:flex!important;
        align-items:center!important;
        gap:10px!important;
        min-width:0!important;
        width:100%!important;
    }

    [data-testid="stExpander"] summary p{
        margin:0!important;
        padding:0!important;
        line-height:1.35!important;
        white-space:normal!important;
        overflow-wrap:anywhere!important;
        word-break:normal!important;
        flex:1 1 auto!important;
        min-width:0!important;
    }

    [data-testid="stExpander"] summary svg,
    [data-testid="stExpander"] summary [data-testid="stIconMaterial"],
    [data-testid="stExpander"] summary span[data-testid="stIconMaterial"]{
        flex:0 0 auto!important;
        width:18px!important;
        min-width:18px!important;
        height:18px!important;
        margin:0!important;
        position:static!important;
        transform:none!important;
    }

    [data-testid="stExpander"] summary [data-testid="stIconMaterial"],
    [data-testid="stExpander"] summary span[data-testid="stIconMaterial"]{
        font-size:18px!important;
        line-height:18px!important;
    }

    /* Prevent any inherited negative margins/absolute positioning from
       custom/global typography rules affecting expander labels. */
    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] summary > *{
        text-indent:0!important;
    }
    [data-testid="stAlert"] p{font-size:14px!important;line-height:1.5!important;}


    /* =========================================================
       FINAL EXPANDER OVERRIDE
       Clean arrow + visible title text, with no overlap.
       ========================================================= */
    [data-testid="stExpander"] details > summary{
        position:relative!important;
        display:flex!important;
        align-items:center!important;
        min-height:44px!important;
        padding:9px 14px 9px 38px!important;
        box-sizing:border-box!important;
        overflow:visible!important;
        cursor:pointer!important;
    }

    /* Hide only Streamlit's internal icon widgets.
       Do NOT hide generic spans because the label text may live in one. */
    [data-testid="stExpander"] details > summary [data-testid="stIconMaterial"],
    [data-testid="stExpander"] details > summary span[data-testid="stIconMaterial"],
    [data-testid="stExpander"] details > summary svg{
        display:none!important;
    }

    /* Custom disclosure arrow */
    [data-testid="stExpander"] details > summary::before{
        content:"›";
        position:absolute!important;
        left:15px!important;
        top:50%!important;
        transform:translateY(-52%) rotate(0deg)!important;
        width:14px!important;
        height:20px!important;
        line-height:18px!important;
        text-align:center!important;
        font-family:Arial,sans-serif!important;
        font-size:22px!important;
        font-weight:600!important;
        color:#64748B!important;
        transition:transform .15s ease!important;
        pointer-events:none!important;
    }

    [data-testid="stExpander"] details[open] > summary::before{
        transform:translateY(-52%) rotate(90deg)!important;
    }

    /* Preserve every non-icon wrapper so Streamlit's label remains visible. */
    [data-testid="stExpander"] details > summary > div,
    [data-testid="stExpander"] details > summary > span:not([data-testid="stIconMaterial"]){
        display:flex!important;
        align-items:center!important;
        width:100%!important;
        min-width:0!important;
        margin:0!important;
        padding:0!important;
        overflow:visible!important;
    }

    [data-testid="stExpander"] details > summary p{
        display:block!important;
        visibility:visible!important;
        opacity:1!important;
        width:100%!important;
        margin:0!important;
        padding:0!important;
        font-family:var(--app-font)!important;
        font-size:14px!important;
        font-weight:650!important;
        line-height:1.35!important;
        color:#334763!important;
        white-space:normal!important;
        overflow:visible!important;
        text-overflow:clip!important;
        overflow-wrap:break-word!important;
        word-break:normal!important;
        text-indent:0!important;
    }

    /* Ensure label text itself is not accidentally hidden by inherited rules. */
    [data-testid="stExpander"] details > summary span:not([data-testid="stIconMaterial"]),
    [data-testid="stExpander"] details > summary div{
        visibility:visible!important;
        opacity:1!important;
        color:#334763!important;
    }


    /* =========================================================
       GLOBAL READABILITY BOOST
       Slightly larger typography across Buildathon, Experiments
       and Learning Summary while preserving the existing hierarchy.
       ========================================================= */
    .stApp{
        font-size:15px!important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stText"],
    .stApp label,
    .stApp input,
    .stApp textarea,
    .stApp select{
        font-size:15px!important;
        line-height:1.58!important;
    }

    .stApp h1{font-size:36px!important;}
    .stApp h2{font-size:28px!important;}
    .stApp h3{font-size:23px!important;}
    .stApp h4{font-size:17px!important;}

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *{
        font-size:14px!important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] *{
        font-size:24px!important;
        font-weight:750!important;
    }

    [data-testid="stExpander"] summary p{
        font-size:15px!important;
        line-height:1.4!important;
    }

    [data-testid="stAlert"],
    [data-testid="stAlert"] *,
    [data-testid="stAlert"] p{
        font-size:14px!important;
        line-height:1.55!important;
    }

    .stButton button,
    .stFormSubmitButton button,
    [data-testid="stButton"] button,
    [data-testid="stFormSubmitButton"] button{
        font-size:15px!important;
    }

    /* Sidebar readability */
    .side-title{font-size:17px!important;}
    .side-sub{font-size:11px!important;}
    .ref-nav-title{font-size:14px!important;}
    .ref-nav-sub{font-size:11px!important;}
    .side-kicker{font-size:11px!important;}
    .side-copy{font-size:13px!important;line-height:1.5!important;}
    .tech-row{font-size:13px!important;}
    .guardrail-row{font-size:12.5px!important;line-height:1.4!important;}
    .eval-side-row{font-size:12.5px!important;line-height:1.4!important;}
    .eval-side-note{font-size:11.5px!important;line-height:1.4!important;}

    /* Custom page components */
    .section-title{font-size:26px!important;}
    .section-sub{font-size:15px!important;line-height:1.55!important;}
    .arch-name{font-size:15px!important;}
    .arch-role{font-size:13px!important;line-height:1.45!important;}
    .exp-overview-title{font-size:24px!important;}
    .exp-overview-heading{font-size:15px!important;}
    .exp-overview-text{font-size:13.5px!important;line-height:1.45!important;}
    .compare-title{font-size:18px!important;}
    .compare-step{font-size:14px!important;line-height:1.45!important;}
    .compare-foot{font-size:13px!important;line-height:1.45!important;}


    /* =========================================================
       LIVE PROCESSING TRACKER
       ========================================================= */
    .live-progress-shell{
        margin:14px 0 16px 0;
        padding:14px 16px 12px 16px;
        border:1px solid #DDE5F5;
        border-radius:12px;
        background:linear-gradient(180deg,#FBFDFF 0%,#F7FAFF 100%);
        box-shadow:0 2px 8px rgba(31,52,96,.04);
    }

    .live-progress-head{
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:12px;
        margin-bottom:13px;
    }

    .live-progress-kicker{
        display:block;
        font-size:10px!important;
        font-weight:800!important;
        letter-spacing:.13em!important;
        color:#6B7A98!important;
        text-transform:uppercase;
        margin-bottom:2px;
    }

    .live-progress-title{
        display:block;
        font-size:15px!important;
        font-weight:750!important;
        color:#17223F!important;
    }

    .live-progress-badge{
        display:inline-flex;
        align-items:center;
        justify-content:center;
        min-height:25px;
        padding:4px 10px;
        border-radius:999px;
        font-size:10px!important;
        font-weight:800!important;
        letter-spacing:.04em;
        white-space:nowrap;
    }

    .live-progress-badge.idle{
        color:#60708D!important;
        background:#EEF2F8;
    }

    .live-progress-badge.running{
        color:#3548D9!important;
        background:#E9EDFF;
        animation:liveBadgePulse 1.35s ease-in-out infinite;
    }

    .live-progress-badge.complete{
        color:#218447!important;
        background:#E8F7EC;
    }

    .live-progress-badge.cached{
        color:#315ED3!important;
        background:#EAF1FF;
    }

    .live-progress-badge.blocked{
        color:#B52E43!important;
        background:#FCE9EC;
    }

    .live-progress-track{
        display:flex;
        align-items:flex-start;
        width:100%;
        gap:0;
    }

    .live-stage{
        flex:0 0 94px;
        min-width:78px;
        display:flex;
        flex-direction:column;
        align-items:center;
        text-align:center;
        position:relative;
    }

    .live-stage-dot{
        width:25px;
        height:25px;
        border-radius:50%;
        border:2px solid #CAD4E6;
        background:#FFFFFF;
        display:flex;
        align-items:center;
        justify-content:center;
        position:relative;
        z-index:2;
        transition:all .2s ease;
    }

    .live-stage-check{
        opacity:0;
        color:#FFFFFF!important;
        font-size:12px!important;
        font-weight:900!important;
        line-height:1!important;
    }

    .live-stage-label{
        margin-top:6px;
        font-size:11.5px!important;
        font-weight:650!important;
        line-height:1.25!important;
        color:#76839B!important;
        white-space:normal;
    }

    .live-stage.done .live-stage-dot{
        border-color:#4F66F3;
        background:#4F66F3;
    }

    .live-stage.done .live-stage-check{
        opacity:1;
    }

    .live-stage.done .live-stage-label{
        color:#334763!important;
    }

    .live-stage.active .live-stage-dot{
        border-color:#4F66F3;
        background:#FFFFFF;
        box-shadow:0 0 0 5px rgba(79,102,243,.11);
        animation:liveDotPulse 1.15s ease-in-out infinite;
    }

    .live-stage.active .live-stage-dot::after{
        content:"";
        width:8px;
        height:8px;
        border-radius:50%;
        background:#4F66F3;
    }

    .live-stage.active .live-stage-label{
        color:#3146D8!important;
        font-weight:800!important;
    }

    .live-stage.blocked .live-stage-dot{
        border-color:#E84F63;
        background:#E84F63;
    }

    .live-stage.blocked .live-stage-dot::after{
        content:"×";
        color:#FFFFFF;
        font-size:16px;
        line-height:1;
        font-weight:800;
    }

    .live-stage.blocked .live-stage-label{
        color:#B52E43!important;
        font-weight:800!important;
    }

    .live-stage-line{
        flex:1 1 auto;
        height:3px;
        margin-top:11px;
        min-width:18px;
        border-radius:999px;
        background:#E4E9F2;
        position:relative;
        overflow:hidden;
    }

    .live-stage-line.done{
        background:#AFC0FF;
    }

    .live-stage-line.active{
        background:#DDE4FF;
    }

    .live-stage-line.active span{
        display:block;
        position:absolute;
        inset:0;
        width:42%;
        border-radius:999px;
        background:linear-gradient(
            90deg,
            rgba(79,102,243,0) 0%,
            rgba(79,102,243,.95) 48%,
            rgba(79,102,243,0) 100%
        );
        animation:liveLineSweep 1.15s linear infinite;
    }

    .live-progress-message{
        margin-top:9px;
        min-height:18px;
        text-align:center;
        font-size:11.5px!important;
        line-height:1.35!important;
        color:#6A7893!important;
    }

    @keyframes liveDotPulse{
        0%,100%{
            box-shadow:0 0 0 4px rgba(79,102,243,.10);
            transform:scale(1);
        }
        50%{
            box-shadow:0 0 0 8px rgba(79,102,243,.05);
            transform:scale(1.06);
        }
    }

    @keyframes liveLineSweep{
        0%{ transform:translateX(-120%); }
        100%{ transform:translateX(340%); }
    }

    @keyframes liveBadgePulse{
        0%,100%{ opacity:.78; }
        50%{ opacity:1; }
    }

    @media(max-width:1100px){
      [data-testid="stSidebar"]{min-width:240px!important;max-width:240px!important;width:240px!important}
      [data-testid="stSidebar"]>div:first-child{width:240px!important}
      .hero-grid{grid-template-columns:1fr}.hero-quote{display:none}.arch-flow{grid-template-columns:1fr}.arch-arrow{transform:rotate(90deg)}.story-shell{flex-direction:column;align-items:stretch}.story-split{grid-template-columns:1fr}
    }
    </style>
    <div class="enterprise-hero">
      <div class="hero-grid">
        <div>
          <div class="eyebrow">Applied AI Agents + Learning Extensions</div>
          <div class="hero-title">AI Customer Support Lab</div>
          <div class="hero-copy">Buildathon-compliant three-agent CrewAI customer support, extended with controlled experiments that test when agentic orchestration adds value over explicit Python workflows.</div>
          <span class="tag">CrewAI</span><span class="tag">Streamlit</span><span class="tag">Guardrails</span><span class="tag">Evals</span><span class="tag">API Efficiency</span>
        </div>
        <div class="hero-quote">“Agents are not always required.<br>But when the path is dynamic,<br>they can add real value.”<small>— Build · Experiment · Learn</small></div>
      </div>
    </div>
    """, unsafe_allow_html=True,

    
)

missing_keys = validate_configuration()
if missing_keys:
    st.error("Missing environment variables: " + ", ".join(missing_keys)); st.stop()

# Sidebar navigation uses query parameters so the reference icons can be rendered exactly.
_page = st.query_params.get("page", "buildathon")
if isinstance(_page, list):
    _page = _page[0] if _page else "buildathon"
_page = str(_page).lower()
if _page not in {"buildathon", "experiments", "learning"}:
    _page = "buildathon"

navigation = {
    "buildathon": "🏠  Buildathon",
    "experiments": "🧪  Experiments",
    "learning": "📊  Learning Summary",
}[_page]

with st.sidebar:
    st.markdown("""<div class='side-brand'><div class='side-logo brand-bot'><svg viewBox='0 0 48 48' aria-hidden='true'><rect x='9' y='14' width='30' height='25' rx='7'/><rect x='19' y='7' width='10' height='6' rx='3'/><circle cx='18' cy='25' r='2.5'/><circle cx='30' cy='25' r='2.5'/><path d='M18 32h12'/></svg></div><div><div class='side-title'>AI Customer<br>Support Lab</div><div class='side-sub'>Build · Experiment · Learn</div></div></div>""", unsafe_allow_html=True)

    build_active = " active" if _page == "buildathon" else ""
    experiment_active = " active" if _page == "experiments" else ""
    learning_active = " active" if _page == "learning" else ""

    st.markdown(f"""
    <div class='ref-nav'>
      <a class='ref-nav-item{build_active}' href='?page=buildathon' target='_self'>
        <span class='ref-nav-icon'>
          <svg viewBox='0 0 32 32' aria-hidden='true'>
            <path fill='#E62D43' d='M3 14.2 16 3l13 11.2v14.3h-8.1v-8.2h-9.8v8.2H3z'/>
            <path fill='#D9233B' d='M1.6 14.5 16 2l14.4 12.5-2.2 2.6L16 6.6 3.8 17.1z'/>
          </svg>
        </span>
        <span><div class='ref-nav-title'>Buildathon</div><div class='ref-nav-sub'>Three-agent CrewAI implementation</div></span>
      </a>

      <a class='ref-nav-item{experiment_active}' href='?page=experiments' target='_self'>
        <span class='ref-nav-icon'>
          <svg viewBox='0 0 32 32' aria-hidden='true'>
            <path fill='#4864F2' d='M11 2h10v3h-2v7l7.3 11.2A4.25 4.25 0 0 1 22.7 30H9.3a4.25 4.25 0 0 1-3.6-6.8L13 12V5h-2z'/>
            <path fill='#BFCBFF' d='M10.5 22h11l-4-6.1h-3.1z'/>
            <circle cx='18.8' cy='23.4' r='1.4' fill='#FFFFFF' opacity='.92'/>
          </svg>
        </span>
        <span><div class='ref-nav-title'>Experiments</div><div class='ref-nav-sub'>Compare approaches &amp; learn</div></span>
      </a>

      <a class='ref-nav-item{learning_active}' href='?page=learning' target='_self'>
        <span class='ref-nav-icon'>
          <svg viewBox='0 0 32 32' aria-hidden='true'>
            <rect x='4' y='18' width='6' height='10' rx='1' fill='#5F8BFF'/>
            <rect x='13' y='11' width='6' height='17' rx='1' fill='#4778FA'/>
            <rect x='22' y='4' width='6' height='24' rx='1' fill='#3565EE'/>
          </svg>
        </span>
        <span><div class='ref-nav-title'>Learning Summary</div><div class='ref-nav-sub'>Key insights and takeaways</div></span>
      </a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='side-kicker'>Project Context</div><div class='side-copy'>Customer support automation using CrewAI agents, with controlled experiments to understand when agentic orchestration adds value over explicit Python workflows.</div>", unsafe_allow_html=True)
    st.markdown("<div class='side-kicker'>Tech Stack</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='tech-stack'>
      <div class='tech-row'>
        <span class='tech-icon'>
          <svg viewBox='0 0 32 32' aria-hidden='true'>
            <path fill='#E9573F' d='M3 14 16 3l13 11v15H3z'/>
            <path fill='#F4C34B' d='M2 14.2 16 2l14 12.2-2 2.5L16 6.3 4 16.7z'/>
            <rect x='7' y='14' width='18' height='14' rx='1.5' fill='#F6F0DA'/>
            <rect x='13' y='19' width='6' height='9' fill='#2F6FD8'/>
            <rect x='8' y='26.5' width='16' height='2.5' rx='1.2' fill='#40A96B'/>
          </svg>
        </span><span>CrewAI</span>
      </div>
      <div class='tech-row'>
        <span class='tech-icon'>
          <svg viewBox='0 0 32 32' aria-hidden='true'>
            <path fill='#EE3E5B' d='M3 10.5h26L24.5 28h-17z'/>
            <path fill='#FFFFFF' opacity='.97' d='m8 11 8 8 8-8z'/>
            <path fill='#D92544' d='m6.2 9.5 4-6.3 5.8 6.3zM15.9 9.5l4-6.3 5.8 6.3z'/>
          </svg>
        </span><span>Streamlit</span>
      </div>
      <div class='tech-row'>
        <span class='tech-icon'>
          <svg viewBox='0 0 32 32' aria-hidden='true'>
            <g fill='none' stroke='#35A76F' stroke-width='2.4' stroke-linecap='round'>
              <path d='M15.7 4.2c3.9-2.2 8.8.5 8.8 5v3.1'/><path d='M24.5 12.3c4.1 1.4 5.2 6.7 2 9.5l-2.4 2.1'/><path d='M24.1 23.9c-.9 4.3-5.9 6.1-9.2 3.4l-2.5-2'/><path d='M12.4 25.3c-3.2 3-8.5 1.1-9.2-3.2l-.5-3.1'/><path d='M2.7 19c-3.2-2.9-1.9-8.1 2.3-9.4l3-.9'/><path d='M8 8.7C7.9 4.4 12.4 1.5 16 3.8l2.6 1.6'/>
              <path d='M9.4 10.4 16 6.7l6.6 3.7v7.3L16 21.3l-6.6-3.6z'/>
            </g>
          </svg>
        </span><span>OpenAI</span>
      </div>
      <div class='tech-row'>
        <span class='tech-icon'>
          <svg viewBox='0 0 32 32' aria-hidden='true'><circle cx='13.5' cy='13.5' r='8' fill='none' stroke='#3DA578' stroke-width='3.2'/><path d='m19.3 19.3 8 8' stroke='#3DA578' stroke-width='3.5' stroke-linecap='round'/></svg>
        </span><span>Web Search</span>
      </div>
      <div class='tech-row'>
        <span class='tech-icon'>
          <svg viewBox='0 0 32 32' aria-hidden='true'>
            <path fill='#3776AB' d='M16 3c-6.2 0-5.8 2.7-5.8 2.7v4h5.9V11H8s-4-.5-4 5.9 3.5 6.1 3.5 6.1h2.1v-3c0-3.4 3-5.1 5.9-5.1h5.8c3 0 5.4-2.4 5.4-5.4V6.1S27.2 3 16 3z'/>
            <circle cx='13.4' cy='6.8' r='1.3' fill='#fff'/>
            <path fill='#FFD343' d='M16 29c6.2 0 5.8-2.7 5.8-2.7v-4h-5.9V21H24s4 .5 4-5.9-3.5-6.1-3.5-6.1h-2.1v3c0 3.4-3 5.1-5.9 5.1h-5.8c-3 0-5.4 2.4-5.4 5.4v3.4S4.8 29 16 29z'/>
            <circle cx='18.6' cy='25.2' r='1.3' fill='#fff'/>
          </svg>
        </span><span>Python</span>
      </div>
      <div class='tech-row'>
        <span class='tech-icon'>
          <svg viewBox='0 0 32 32' aria-hidden='true'><path d='M8 3h12l5 5v21H8z' fill='none' stroke='#8A3F2E' stroke-width='2.7'/><path d='M20 3v6h6M12 15h9M12 20h9M12 25h7' fill='none' stroke='#8A3F2E' stroke-width='2.2' stroke-linecap='round'/></svg>
        </span><span>Text Persistence (.txt)</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


    if _page == "learning":
        st.markdown(
            "<div class='side-kicker'>Learning Lens</div>"
            "<div class='eval-side-panel'>"
            "<div class='eval-side-head'>"
            "<span class='eval-side-title'>What this lab proves</span>"
            "<span class='eval-side-badge'>LEARN</span>"
            "</div>"
            "<div class='guardrail-row'><span class='ok'>✓</span><span>Agents are not automatically required</span></div>"
            "<div class='guardrail-row'><span class='ok'>✓</span><span>Fixed paths favor explicit orchestration</span></div>"
            "<div class='guardrail-row'><span class='ok'>✓</span><span>Dynamic paths benefit from runtime selection</span></div>"
            "<div class='guardrail-row'><span class='ok'>✓</span><span>Grounding must state what evidence it uses</span></div>"
            "<div class='guardrail-row'><span class='ok'>✓</span><span>Validation and API efficiency are architecture concerns</span></div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        # Latest security evidence: visible below Tech Stack.
        latest_security = (
            st.session_state.get("buildathon_security")
            if _page == "buildathon"
            else st.session_state.get("security")
        )
        st.markdown("<div class='side-kicker'>Guardrails</div>", unsafe_allow_html=True)
        if latest_security:
            det = latest_security.get("deterministic") or {}
            checks = det.get("checks", [])
            allowed = bool(latest_security.get("allowed"))
            badge_class = "pass" if allowed else "block"
            badge_text = "PASS" if allowed else "BLOCKED"
            rows = []
            for check in checks:
                passed = check.get("status") == "PASS"
                symbol = "✓" if passed else "✕"
                symbol_class = "ok" if passed else "bad"
                rows.append(
                    f"<div class='guardrail-row'><span class='{symbol_class}'>{symbol}</span>"
                    f"<span>{check.get('name','Check')}</span></div>"
                )
            ai = latest_security.get("ai_safety")
            if ai:
                ai_status = "SAFE" if ai.get("allowed") else "BLOCKED"
                ai_category = str(ai.get("category", "")).strip()
                ai_text = ai_status + (f" · {ai_category}" if ai_category else "")
            else:
                ai_text = "Not called — deterministic guardrail stopped the request"
            latest_output_guardrail = (
                (st.session_state.get("buildathon_result") or {})
                .get("output_guardrail")
                if _page == "buildathon"
                else None
            )

            output_html = ""
            if latest_output_guardrail:
                output_status = (
                    "PASS"
                    if latest_output_guardrail.get("passed")
                    else "BLOCKED"
                )
                output_html = (
                    "<div class='guardrail-ai'>"
                    "<strong>Output Guardrail</strong><br>"
                    f"{output_status} · deterministic post-generation gate"
                    "</div>"
                )

            st.markdown(
                "<div class='guardrail-panel'>"
                "<div class='guardrail-head'><span class='guardrail-title'>Latest Run</span>"
                f"<span class='guardrail-badge {badge_class}'>{badge_text}</span></div>"
                + "".join(rows)
                + f"<div class='guardrail-ai'><strong>AI Safety</strong><br>{ai_text}</div>"
                + output_html
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='guardrail-panel'>"
                "<div class='guardrail-head'><span class='guardrail-title'>Latest Run</span>"
                "<span class='guardrail-badge pending'>NOT RUN</span></div>"
                "<div class='guardrail-row'><span>○</span><span>Run the Buildathon workflow to display security evidence.</span></div></div>",
                unsafe_allow_html=True,
            )


        # Latest evaluation evidence.
        # Buildathon Core does NOT execute a new independent evaluation search.
        # This panel only displays the latest evaluation already produced by the
        # Experiments / learning-extension evaluation flow.
        latest_eval = (
            st.session_state.get("buildathon_evaluation")
            if _page == "buildathon"
            else (
                st.session_state.get("latest_evaluation")
                or st.session_state.get("evaluation")
            )
        )

        st.markdown("<div class='side-kicker'>Evals</div>", unsafe_allow_html=True)

        if latest_eval and latest_eval.get("status") == "SUCCESS":
            eval_result = latest_eval.get("result") or {}
            evaluation_mode = latest_eval.get("evaluation_mode")

            temporal_eval = latest_eval.get("temporal_validation") or {}
            temporal_html = ""
            if temporal_eval.get("applied"):
                temporal_html = (
                    "<div class='eval-side-row'><span>Temporal</span><strong>"
                    + str(temporal_eval.get("status", "—"))
                    + "</strong></div>"
                )

            # Experiment 2 is a true side-by-side comparison, so show BOTH
            # Explicit Python and Runtime Agent scores in the sidebar.
            if evaluation_mode == "tool_trace":
                python_scores = eval_result.get("non_agentic") or {}
                agent_scores = eval_result.get("agentic") or {}

                def _paired_eval_score(metric_name):
                    python_metric = python_scores.get(metric_name) or {}
                    agent_metric = agent_scores.get(metric_name) or {}

                    python_score = (
                        python_metric.get("score", "—")
                        if isinstance(python_metric, dict)
                        else "—"
                    )
                    agent_score = (
                        agent_metric.get("score", "—")
                        if isinstance(agent_metric, dict)
                        else "—"
                    )

                    return f"{python_score} / {agent_score}"

                sidebar_eval_html = (
                    "<div class='eval-side-panel'>"
                    "<div class='eval-side-head'>"
                    "<span class='eval-side-title'>Latest Evaluation</span>"
                    "<span class='eval-side-badge'>LATEST</span>"
                    "</div>"
                    "<div class='eval-side-note' style='padding-top:0;padding-bottom:5px;'>"
                    "<strong>Python / Runtime Agent</strong>"
                    "</div>"
                    f"<div class='eval-side-row'><span>Relevance</span><strong>{_paired_eval_score('relevance')}</strong></div>"
                    f"<div class='eval-side-row'><span>Completeness</span><strong>{_paired_eval_score('completeness')}</strong></div>"
                    f"<div class='eval-side-row'><span>Consistency</span><strong>{_paired_eval_score('consistency')}</strong></div>"
                    f"<div class='eval-side-row'><span>Trace Grounding</span><strong>{_paired_eval_score('groundedness')}</strong></div>"
                    "<div class='eval-side-note'>"
                    "Experiment 2 compares both responses against the customer request "
                    "and their actual simulated execution traces. Trace Grounding checks "
                    "whether claimed support actions are backed by those traces. "
                    "No web search is used for this evaluation."
                    "</div>"
                    "</div>"
                )

            else:
                # Buildathon and Experiment 1 continue to display the most
                # relevant single score set in the compact sidebar.
                eval_scores = (
                    eval_result.get("agentic")
                    or eval_result.get("non_agentic")
                    or eval_result
                )

                def _eval_score(metric_name):
                    metric = eval_scores.get(metric_name) or {}
                    return (
                        metric.get("score", "—")
                        if isinstance(metric, dict)
                        else "—"
                    )

                if _page == "buildathon":
                    grounding_label = "Workflow Grounding"
                    eval_note = (
                        "Buildathon grounding checks the final answer against the same raw "
                        "workflow evidence already used for generation. It is not a second "
                        "independent fact-check."
                    )
                else:
                    grounding_label = "Independent Groundedness"
                    eval_note = (
                        "Experiment 1 groundedness is scored against a separate independent "
                        "evaluation evidence set."
                    )

                sidebar_eval_html = (
                    "<div class='eval-side-panel'>"
                    "<div class='eval-side-head'>"
                    "<span class='eval-side-title'>Latest Evaluation</span>"
                    "<span class='eval-side-badge'>LATEST</span>"
                    "</div>"
                    f"<div class='eval-side-row'><span>Relevance</span><strong>{_eval_score('relevance')}/100</strong></div>"
                    f"<div class='eval-side-row'><span>Completeness</span><strong>{_eval_score('completeness')}/100</strong></div>"
                    f"<div class='eval-side-row'><span>Consistency</span><strong>{_eval_score('consistency')}/100</strong></div>"
                    f"<div class='eval-side-row'><span>{grounding_label}</span><strong>{_eval_score('groundedness')}/100</strong></div>"
                    + temporal_html
                    + f"<div class='eval-side-note'>{eval_note}</div>"
                    "</div>"
                )

            st.markdown(
                sidebar_eval_html,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='eval-side-panel'>"
                "<div class='eval-side-head'>"
                "<span class='eval-side-title'>Latest Evaluation</span>"
                "<span class='guardrail-badge pending'>NOT RUN</span>"
                "</div>"
                "<div class='eval-side-note'>Experiment 1 uses independent answer-quality evaluation. Experiment 2 uses deterministic decision/tool-trace validation instead.</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        dynamic_validation_side = st.session_state.get("dynamic_decision_validation")
        if _page == "experiments" and dynamic_validation_side:
            side_status = dynamic_validation_side.get("status", "FAIL")
            side_badge = "pass" if side_status == "PASS" else "block"
            agreement = dynamic_validation_side.get("selection_agreement", 0)
            total_caps = dynamic_validation_side.get("capability_count", 4)
            avoided = dynamic_validation_side.get("agent_tools_avoided", 0)

            st.markdown(
                "<div class='side-kicker'>Decision Validation</div>"
                "<div class='guardrail-panel'>"
                "<div class='guardrail-head'>"
                "<span class='guardrail-title'>Latest Run</span>"
                f"<span class='guardrail-badge {side_badge}'>{side_status}</span>"
                "</div>"
                f"<div class='eval-side-row'><span>Selection agreement</span><strong>{agreement}/{total_caps}</strong></div>"
                f"<div class='eval-side-row'><span>Agent tools avoided</span><strong>{avoided}/{total_caps}</strong></div>"
                "<div class='eval-side-note'>Uses actual routing and tool traces. "
                "Adds zero OpenAI/search calls.</div>"
                "</div>",
                unsafe_allow_html=True,
            )

experiment = ""
query = ""
run_button = False

if navigation == "📊  Learning Summary":
    st.markdown(
        "<div class='section-title'>📊 Learning Summary</div>"
        "<div class='section-sub'>What the buildathon and controlled experiments "
        "demonstrated about agentic orchestration, grounding, validation and API efficiency.</div>",
        unsafe_allow_html=True,
    )

    st.info(
        "**Core learning:** Agents are not automatically better. Their value increases "
        "when the execution path varies, context must be interpreted at runtime, and "
        "the system must choose among bounded capabilities. For a fixed, predictable "
        "path, explicit Python orchestration can be simpler and faster."
    )

    # ------------------------------------------------------------
    # Three-part learning journey
    # ------------------------------------------------------------
    st.markdown("### Learning Journey")
    journey_cols = st.columns(3, gap="large")

    with journey_cols[0]:
        with st.container(border=True):
            st.markdown("#### 🏠 Buildathon Core")
            st.caption("Build the required agentic workflow correctly")
            st.markdown(
                "- Three specialized CrewAI agents\n"
                "- Sequential hand-offs\n"
                "- Current web evidence\n"
                "- Final grounded synthesis\n"
                "- Guardrails + evaluation\n"
                "- Text persistence"
            )
            st.success("Focus: build a complete, controlled agent workflow.")

    with journey_cols[1]:
        with st.container(border=True):
            st.markdown("#### 🧩 Experiment 1")
            st.caption("Fixed Sequential Workflow")
            st.markdown(
                "- Same predictable request path\n"
                "- Explicit Python vs CrewAI\n"
                "- Shared workflow evidence\n"
                "- Independent evaluation evidence\n"
                "- Quality vs latency/orchestration overhead"
            )
            st.info("Learning: agents may add overhead when the path is already known.")

    with journey_cols[2]:
        with st.container(border=True):
            st.markdown("#### 🧭 Experiment 2")
            st.caption("Dynamic Decision Workflow")
            st.markdown(
                "- Variable request paths\n"
                "- Four bounded capabilities\n"
                "- Rules vs runtime tool selection\n"
                "- Tool-trace grounding\n"
                "- Decision/action validation"
            )
            st.info("Learning: runtime selection becomes useful when the path varies.")

    # ------------------------------------------------------------
    # Decision guide
    # ------------------------------------------------------------
    st.markdown("### When to Use Explicit Python vs Agents")
    st.caption(
        "The practical decision is not 'agents or no agents?' — it is where the "
        "decision logic should live."
    )

    guide_left, guide_right = st.columns(2, gap="large")

    with guide_left:
        with st.container(border=True):
            st.markdown("### 🐍 Prefer Explicit Python")
            st.markdown(
                "**Best fit when:**\n"
                "- execution sequence is fixed and predictable\n"
                "- routing conditions are few and stable\n"
                "- deterministic behavior is the priority\n"
                "- latency and cost should stay minimal\n"
                "- every branch can be clearly encoded and tested"
            )
            st.success("Decision ownership: developer-authored rules.")

    with guide_right:
        with st.container(border=True):
            st.markdown("### 🤖 Prefer Bounded Agentic Orchestration")
            st.markdown(
                "**Best fit when:**\n"
                "- execution path varies by request\n"
                "- multiple capabilities may be combined dynamically\n"
                "- intent/context must be interpreted at runtime\n"
                "- tool relevance cannot be fully predetermined\n"
                "- flexibility is worth additional orchestration overhead"
            )
            st.success("Decision ownership: runtime reasoning within tool boundaries.")

    # ------------------------------------------------------------
    # Grounding / validation learning
    # ------------------------------------------------------------
    st.markdown("### Grounding & Validation — What I Learned")

    g1, g2, g3 = st.columns(3, gap="large")

    with g1:
        with st.container(border=True):
            st.markdown("#### Workflow Grounding")
            st.caption("Used in Buildathon")
            st.write(
                "Checks whether the final answer is supported by the evidence "
                "already retrieved for that workflow."
            )
            st.markdown("**Question answered:** *Did we use our evidence correctly?*")

    with g2:
        with st.container(border=True):
            st.markdown("#### Independent Groundedness")
            st.caption("Used in Experiment 1")
            st.write(
                "Checks both generated answers against a separate independent "
                "evidence set retrieved after generation."
            )
            st.markdown("**Question answered:** *Does separate evidence support the claim?*")

    with g3:
        with st.container(border=True):
            st.markdown("#### Trace Grounding")
            st.caption("Used in Experiment 2")
            st.write(
                "Checks whether operational claims are backed by the actual "
                "simulated rule/tool execution trace."
            )
            st.markdown("**Question answered:** *Did the claimed action really execute?*")

    st.info(
        "**Important distinction:** two answers agreeing with each other does not "
        "prove that they are factually correct. Grounding must always state what "
        "evidence or execution trace the answer is being checked against."
    )

    # ------------------------------------------------------------
    # Guardrails & Evals production learning
    # ------------------------------------------------------------
    st.markdown("### Guardrails & Evals — Production Lessons")
    st.caption(
        "A successful workflow execution is not enough. Production-style AI needs "
        "controls before generation, controls before release, and quality measurement "
        "after generation."
    )

    ge1, ge2, ge3 = st.columns(3, gap="large")

    with ge1:
        with st.container(border=True):
            st.markdown("#### 🛡 Input Guardrails")
            st.caption("Before expensive workflow execution")
            st.markdown(
                "- Query validation\n"
                "- Input-length limits\n"
                "- Prompt-injection checks\n"
                "- Secret-extraction protection\n"
                "- AI safety classification"
            )
            st.success(
                "Purpose: stop unsafe or malformed requests before agent/model work begins."
            )

    with ge2:
        with st.container(border=True):
            st.markdown("#### 🛡 Output Guardrail")
            st.caption("After generation · before persistence/release")
            st.markdown(
                "- Output structure validation\n"
                "- Secret/credential leakage checks\n"
                "- Internal-context leakage checks\n"
                "- Unsupported action-claim checks\n"
                "- Temporal safety checks"
            )
            st.success(
                "Purpose: only approved generated output is persisted or released."
            )

    with ge3:
        with st.container(border=True):
            st.markdown("#### ✅ Evals")
            st.caption("Measure quality after generation")
            st.markdown(
                "- Relevance\n"
                "- Completeness\n"
                "- Consistency\n"
                "- Context-appropriate grounding\n"
                "- Clear evaluator evidence"
            )
            st.success(
                "Purpose: measure answer quality rather than assuming success from execution alone."
            )

    st.markdown("#### How the controls fit together")
    control_flow = st.columns(5)

    control_items = [
        ("1", "Input Guardrails", "Can this request enter the workflow?"),
        ("2", "Generation / Agents", "Produce the answer or execute bounded actions."),
        ("3", "Output Guardrail", "Can this generated output be released?"),
        ("4", "Persistence", "Save only approved output."),
        ("5", "Evals", "How good and grounded was the result?"),
    ]

    for col, (step, title, body) in zip(control_flow, control_items):
        with col:
            with st.container(border=True):
                st.markdown(f"**{step}. {title}**")
                st.caption(body)

    st.info(
        "**Production lesson:** Guardrails and evals solve different problems. "
        "Guardrails control what may enter and leave the system; evals measure the "
        "quality of what the system produced. Both are needed."
    )

    # ------------------------------------------------------------
    # Engineering practices
    # ------------------------------------------------------------
    st.markdown("### Engineering Practices Demonstrated")
    practice_cols = st.columns(3, gap="large")

    practices = [
        (
            "🛡 Guardrails",
            "Input controls protect workflow entry; deterministic output controls protect release and persistence.",
        ),
        (
            "✅ Evals",
            "Relevance, completeness, consistency and context-appropriate grounding rather than one generic score.",
        ),
        (
            "🔎 Evidence Reuse",
            "Retrieve once where appropriate and reuse evidence for generation/evaluation instead of duplicate searches.",
        ),
        (
            "🧾 Persistence",
            "Persist the final support record only after structural validation and the Output Guardrail succeed.",
        ),
        (
            "🧪 Deterministic Validation",
            "Sequence, persistence, temporal and tool-trace checks add confidence without another LLM call.",
        ),
        (
            "⚡ API Efficiency",
            "Exact-session reuse, evidence caching and early guardrail blocking reduce unnecessary external calls.",
        ),
    ]

    for index, (title, body) in enumerate(practices):
        with practice_cols[index % 3]:
            with st.container(border=True):
                st.markdown(f"#### {title}")
                st.write(body)

    # ------------------------------------------------------------
    # Measured evidence from this session, if available.
    # ------------------------------------------------------------
    st.markdown("### Measured Evidence from This Session")
    st.caption(
        "This section reads existing session results only. Opening Learning Summary "
        "does not trigger OpenAI or search calls."
    )

    build_cache = _latest_cached_value("buildathon_run_cache")
    exp1_cache = _latest_cached_value("experiment1_run_cache")
    exp2_cache = _latest_cached_value("experiment2_run_cache")

    evidence_cols = st.columns(3, gap="large")

    with evidence_cols[0]:
        with st.container(border=True):
            st.markdown("#### Buildathon")
            if build_cache:
                result = build_cache.get("result") or {}
                evaluation = build_cache.get("evaluation") or {}
                eval_scores = (
                    (evaluation.get("result") or {}).get("agentic") or {}
                )
                st.success("TESTED THIS SESSION")
                st.metric("Agents", result.get("agents", 3))

                build_security = build_cache.get("security") or {}
                input_guardrail_status = (
                    "PASS" if build_security.get("allowed") else "BLOCKED"
                )
                output_guardrail = result.get("output_guardrail") or {}
                output_guardrail_status = (
                    "PASS" if output_guardrail.get("passed") else "BLOCKED"
                )

                b1, b2 = st.columns(2)
                b1.metric("Input Guardrails", input_guardrail_status)
                b2.metric("Output Guardrail", output_guardrail_status)

                st.metric(
                    "Workflow Grounding",
                    f"{(eval_scores.get('groundedness') or {}).get('score', '—')}/100",
                )
                st.caption(
                    "Three-agent sequential workflow with evidence reuse, "
                    "input/output controls and post-generation evaluation."
                )
            else:
                st.info("Run the Buildathon workflow to populate measured evidence.")

    with evidence_cols[1]:
        with st.container(border=True):
            st.markdown("#### Experiment 1")
            if exp1_cache:
                state = exp1_cache.get("state") or {}
                python_result = state.get("non_agentic") or {}
                agent_result = state.get("agentic") or {}
                evaluation = state.get("evaluation") or {}
                eval_result = evaluation.get("result") or {}
                py_ground = (
                    (eval_result.get("non_agentic") or {})
                    .get("groundedness", {})
                    .get("score", "—")
                )
                ag_ground = (
                    (eval_result.get("agentic") or {})
                    .get("groundedness", {})
                    .get("score", "—")
                )
                st.success("TESTED THIS SESSION")
                st.metric(
                    "Execution Time",
                    f"{python_result.get('elapsed_time', 0):.2f}s / "
                    f"{agent_result.get('elapsed_time', 0):.2f}s",
                )
                st.caption("Python / CrewAI")
                st.metric(
                    "Independent Groundedness",
                    f"{py_ground} / {ag_ground}",
                )
                st.caption("Python / CrewAI")
            else:
                st.info("Run Experiment 1 to populate measured evidence.")

    with evidence_cols[2]:
        with st.container(border=True):
            st.markdown("#### Experiment 2")
            if exp2_cache:
                state = exp2_cache.get("state") or {}
                validation = state.get("dynamic_decision_validation") or {}
                evaluation = state.get("dynamic_evaluation") or {}
                eval_result = evaluation.get("result") or {}
                py_trace = (
                    (eval_result.get("non_agentic") or {})
                    .get("groundedness", {})
                    .get("score", "—")
                )
                ag_trace = (
                    (eval_result.get("agentic") or {})
                    .get("groundedness", {})
                    .get("score", "—")
                )

                st.success("TESTED THIS SESSION")
                st.metric(
                    "Selection Agreement",
                    f"{validation.get('selection_agreement', '—')}/"
                    f"{validation.get('capability_count', 4)}",
                )
                st.metric(
                    "Trace Grounding",
                    f"{py_trace} / {ag_trace}",
                )
                st.caption("Python / Runtime Agent")
                st.metric(
                    "Decision Validation",
                    validation.get("status", "—"),
                )
            else:
                st.info("Run Experiment 2 to populate measured evidence.")

    # ------------------------------------------------------------
    # Final takeaways
    # ------------------------------------------------------------
    st.markdown("### Final Takeaways")

    takeaway_left, takeaway_right = st.columns(2, gap="large")

    with takeaway_left:
        with st.container(border=True):
            st.markdown("#### What agents add")
            st.markdown(
                "- runtime interpretation of intent and context\n"
                "- dynamic capability/tool selection\n"
                "- flexible orchestration across variable paths\n"
                "- role separation where it improves maintainability"
            )

    with takeaway_right:
        with st.container(border=True):
            st.markdown("#### What agents do not remove")
            st.markdown(
                "- need for input and output guardrails\n"
                "- need for grounded evidence and evals\n"
                "- need for deterministic validation\n"
                "- need for observability and traceability\n"
                "- need to manage latency, cost and API usage"
            )

    st.success(
        "**Final conclusion:** Use agents deliberately, not by default. "
        "Keep deterministic work deterministic; introduce agentic orchestration "
        "where runtime reasoning and dynamic capability selection create measurable value."
    )

    st.caption(
        "Learning Summary is rendered entirely from local UI content and existing "
        "session state · zero additional OpenAI or search calls. Buildathon "
        "conversation memory also lives only in Streamlit session state and makes "
        "no separate memory API call."
    )


if navigation == "🏠  Buildathon":
    st.markdown("<div class='section-title'>🏠 Buildathon – Three-Agent Customer Support</div><div class='section-sub'>End-to-end implementation using CrewAI with three specialized agents, web search, input/output guardrails and persistent recording.</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='arch-shell'><div class='arch-flow'>
      <div class='arch-card'><div class='arch-icon blue-icon'><svg viewBox='0 0 48 48' aria-hidden='true'><circle cx='20' cy='14' r='7' class='fill-icon'/><circle cx='31' cy='11' r='5' class='fill-icon'/><path class='fill-icon' d='M8 39c0-8 5-13 12-13s12 5 12 13H8z'/><path class='fill-icon' d='M26 37c.3-5 3.7-8.5 8-9.4 3.8 1.5 6 4.7 6 9.4H26z'/></svg></div><div class='arch-name'>Customer Query</div><div class='arch-role'>Natural language<br>support request</div></div><div class='arch-arrow'>→</div>
      <div class='arch-card'><div class='arch-icon guard-icon'><svg viewBox='0 0 48 48' aria-hidden='true'><path class='fill-icon' d='M24 4 39 10v11c0 10.3-6.2 18.3-15 23-8.8-4.7-15-12.7-15-23V10z'/><path d='M24 9v29' stroke='white' stroke-width='3.2'/></svg></div><div class='arch-name'>Security &amp; Guardrails</div><div class='arch-role'>• Input validation<br>• Safety checks<br>• Policy enforcement</div></div><div class='arch-arrow'>→</div>
      <div class='arch-card blue'><div class='arch-icon blue-icon'><svg viewBox='0 0 48 48' aria-hidden='true'><rect x='9' y='14' width='30' height='25' rx='7' class='fill-icon'/><rect x='19' y='6' width='10' height='7' rx='3.5' class='fill-icon'/><path d='M24 4v4' stroke='currentColor' stroke-width='3'/><circle cx='18' cy='25' r='2.4' fill='white'/><circle cx='30' cy='25' r='2.4' fill='white'/><path d='M18 32h12' stroke='white' stroke-width='2.4'/></svg></div><div class='arch-name'>Assistant Agent</div><div class='arch-stage blue-stage'>Generate</div><div class='arch-role'>Interprets the request and provides initial support answer from model knowledge.</div><span class='arch-badge'>No web search</span></div><div class='arch-arrow'>→</div>
      <div class='arch-card green'><div class='arch-icon green-icon'><svg viewBox='0 0 48 48' aria-hidden='true'><circle cx='21' cy='20' r='12' class='fill-icon'/><circle cx='21' cy='20' r='6.5' fill='#EFF9ED'/><path d='m30 29 10 10' stroke='currentColor' stroke-width='6' stroke-linecap='round'/></svg></div><div class='arch-name'>Web Search Assistant</div><div class='arch-stage green-stage'>Enrich</div><div class='arch-role'>Uses web search to produce a current, evidence-informed answer.</div><span class='arch-badge'>Tool enabled</span></div><div class='arch-arrow'>→</div>
      <div class='arch-card purple'><div class='arch-icon purple-icon'><svg viewBox='0 0 48 48' aria-hidden='true'><path d='M14 6h17l7 7v29H14z' fill='none' stroke='currentColor' stroke-width='4'/><path d='M31 6v9h8M20 24h12M20 31h12' fill='none' stroke='currentColor' stroke-width='3'/></svg></div><div class='arch-name'>Entry Agent</div><div class='arch-stage purple-stage'>Consolidate</div><div class='arch-role'>Synthesizes both agent outputs into the final grounded support record.</div><span class='arch-badge'>Final answer</span></div><div class='arch-arrow'>→</div>
      <div class='arch-card output-guard'><div class='arch-icon output-guard-icon'><svg viewBox='0 0 48 48' aria-hidden='true'><path d='M24 4 39 10v11c0 10.3-6.2 18.3-15 23-8.8-4.7-15-12.7-15-23V10z'/><path d='m16.5 24 5 5 10-11' stroke='white' stroke-width='3.4' fill='none'/></svg></div><div class='arch-name'>Output Guardrail</div><div class='arch-stage output-guard-stage'>Validate</div><div class='arch-role'>Checks leakage, internal context, unsupported action claims and temporal safety.</div><span class='arch-badge'>Before persist</span></div><div class='arch-arrow'>→</div>
      <div class='arch-card'><div class='arch-icon blue-icon'><svg viewBox='0 0 48 48' aria-hidden='true'><path d='M14 6h17l7 7v29H14z' fill='none' stroke='currentColor' stroke-width='4'/><path d='M31 6v9h8' fill='none' stroke='currentColor' stroke-width='4'/></svg></div><div class='arch-name'>Persistent Record</div><div class='arch-stage blue-stage'>Save</div><div class='arch-role'>Only approved final output is saved for future reference.</div><span class='arch-badge'>answers.txt</span></div>
    </div></div>""", unsafe_allow_html=True)

    st.markdown("### Buildathon Core")
    st.caption("The required implementation is shown first. Learning experiments are separated so the core submission remains easy to review.")
    c1,c2,c3,c4,c5,c6=st.columns(6)
    c1.metric("CrewAI Agents","3"); c2.metric("Process","Sequential"); c3.metric("Web Search","Yes"); c4.metric("Both Answers","Displayed"); c5.metric("Persistence",".txt"); c6.metric("Secrets","Env Vars")
    st.markdown("<div class='callout'>ℹ️ &nbsp; <b>CrewAI execution model:</b> Sequential — each agent completes its responsibility before handing the workflow to the next agent.</div>", unsafe_allow_html=True)

    # Live execution tracker sits directly below the architecture/core flow.
    buildathon_progress_slot = st.empty()
    saved_progress = st.session_state.get("buildathon_progress_state") or {
        "phase": "idle",
        "message": "Ready to run the guarded three-agent workflow.",
    }
    render_buildathon_progress(
        buildathon_progress_slot,
        saved_progress.get("phase", "idle"),
        saved_progress.get("message", ""),
    )

    st.markdown("### Try it out")

    memory_left, memory_mid, memory_right = st.columns([2.2, 4.8, 1.5])
    with memory_left:
        buildathon_memory_enabled = st.toggle(
            "Session Memory",
            value=True,
            key="buildathon_memory_enabled",
            help=(
                "Keeps the last few completed Buildathon turns in this browser "
                "session so follow-up questions can use prior context."
            ),
        )

    memory_turns = get_session_memory("buildathon")

    with memory_mid:
        if buildathon_memory_enabled:
            st.caption(
                f"Memory ON · {len(memory_turns)}/{SESSION_MEMORY_MAX_TURNS} "
                "recent turn(s) remembered · follow-up references enabled · "
                "local session only · no memory API call"
            )
        else:
            st.caption(
                "Memory OFF · each Buildathon request is treated independently."
            )

    with memory_right:
        if st.button(
            "Clear Memory",
            key="clear_buildathon_memory",
            disabled=not bool(memory_turns),
            use_container_width=True,
        ):
            clear_session_memory("buildathon")
            st.rerun()

    if buildathon_memory_enabled and memory_turns:
        with st.expander("View Session Memory", expanded=False):
            for index, turn in enumerate(memory_turns, start=1):
                st.markdown(f"**Turn {index} · Customer**")
                st.write(turn["user"])
                st.markdown("**Final answer**")
                st.write(turn["assistant"])

    st.caption(
        "Type a customer support question and press Enter, or use the Run button. "
        "Session memory is bounded to the latest completed turns to control token usage."
    )

    with st.form("buildathon_query_form", clear_on_submit=False):
        buildathon_query = st.text_input(
            "Customer Query",
            placeholder="Example: What is the latest stable version of Python?",
            key="buildathon_query",
        )
        buildathon_run = st.form_submit_button(
            "Run Buildathon Workflow",
            type="primary",
            use_container_width=False,
        )
    if buildathon_run:
        # Full-result reuse must include conversational context. The exact same
        # follow-up wording can mean different things after different topics.
        pre_run_memory_context = (
            build_session_memory_context("buildathon")
            if st.session_state.get("buildathon_memory_enabled", True)
            else ""
        )
        buildathon_cache_identity = (
            buildathon_query.strip()
            if not pre_run_memory_context
            else (
                f"{buildathon_query.strip()}\\n"
                f"SESSION MEMORY:\\n{pre_run_memory_context}"
            )
        )

        cached_buildathon = None
        if buildathon_query.strip():
            cached_buildathon, _ = _cache_lookup(
                "buildathon_run_cache",
                buildathon_cache_identity,
                allow_near=False,
            )

        if cached_buildathon is not None:
            cached_result = cached_buildathon.get("result") or {}

            if cached_result.get("output_guardrail") is None:
                cached_result["output_guardrail"] = run_output_guardrail(
                    query=buildathon_query.strip(),
                    assistant_answer=cached_result.get("assistant_answer", ""),
                    web_answer=cached_result.get("web_answer", ""),
                    final_answer=(
                        cached_result.get("final_answer")
                        or cached_result.get("answer", "")
                    ),
                    temporal_validation=cached_result.get(
                        "temporal_validation"
                    ),
                )
                cached_buildathon["result"] = cached_result

            st.session_state["buildathon_progress_state"] = {
                "phase": "cached",
                "message": "Exact session result reused — no new OpenAI or search calls.",
            }
            render_buildathon_progress(
                buildathon_progress_slot,
                "cached",
                "Exact session result reused — no new OpenAI or search calls.",
            )
            clear_run_state()
            st.session_state["buildathon_result"] = cached_buildathon["result"]
            st.session_state["buildathon_evaluation"] = cached_buildathon["evaluation"]
            st.session_state["latest_evaluation"] = cached_buildathon["evaluation"]
            st.session_state["buildathon_security"] = cached_buildathon["security"]
            st.session_state["buildathon_elapsed"] = cached_buildathon.get("elapsed", 0.0)
            st.session_state["buildathon_notice"] = (
                "Reused the exact Buildathon result from this session — no new "
                "OpenAI or search API calls were required."
            )
            st.rerun()

        clear_run_state()
        st.session_state.pop("buildathon_result", None)
        st.session_state.pop("buildathon_evaluation", None)

        # Clear previous Buildathon security evidence immediately so the
        # sidebar can never display a stale result from an earlier run.
        st.session_state.pop("buildathon_security", None)

        if not buildathon_query.strip():
            st.session_state["buildathon_progress_state"] = {
                "phase": "idle",
                "message": "Enter a customer request to start the workflow.",
            }
            render_buildathon_progress(
                buildathon_progress_slot,
                "idle",
                "Enter a customer request to start the workflow.",
            )
            st.error("Please enter a customer query.")
        else:
            total_start = time.perf_counter()

            # Buildathon Core Step 1: deterministic + AI safety guardrails.
            st.session_state["buildathon_progress_state"] = {
                "phase": "guardrails",
                "message": "Validating input, prompt safety and request policy.",
            }
            render_buildathon_progress(
                buildathon_progress_slot,
                "guardrails",
                "Validating input, prompt safety and request policy.",
            )

            with st.spinner("Running security guardrails..."):
                security = run_security_gate(buildathon_query.strip())

            # Save ONLY the security result for this current Buildathon run.
            st.session_state["buildathon_security"] = security

            if not security["allowed"]:
                st.session_state["buildathon_elapsed"] = time.perf_counter() - total_start
                deterministic = security.get("deterministic") or {}
                failed = [
                    c.get("name", "Guardrail")
                    for c in deterministic.get("checks", [])
                    if c.get("status") != "PASS"
                ]
                ai_safety = security.get("ai_safety")
                reason = ", ".join(failed) if failed else (
                    (ai_safety or {}).get("reason")
                    or (ai_safety or {}).get("category")
                    or "Security policy"
                )
                st.session_state["buildathon_progress_state"] = {
                    "phase": "blocked",
                    "message": f"Stopped at Guardrails — {reason}.",
                }
                render_buildathon_progress(
                    buildathon_progress_slot,
                    "blocked",
                    f"Stopped at Guardrails — {reason}.",
                )
                st.session_state["buildathon_block_notice"] = (
                    f"Request blocked by guardrails: {reason}. "
                    "The three-agent workflow was not executed."
                )
                st.rerun()
            else:
                safe_query = security["deterministic"]["query"]

                buildathon_memory_context = (
                    build_session_memory_context("buildathon")
                    if st.session_state.get("buildathon_memory_enabled", True)
                    else ""
                )
                buildathon_search_context = (
                    build_search_memory_context("buildathon")
                    if st.session_state.get("buildathon_memory_enabled", True)
                    else ""
                )

                # Buildathon Core Step 2:
                # Run ONLY the required three-agent CrewAI workflow.
                #
                # Assistant Agent -> Web Search Assistant -> Entry Agent
                #
                # The Web Search Assistant inside run_agentic() is the search
                # capability required by the Buildathon. No second independent
                # evaluation search is executed here.
                st.session_state["buildathon_progress_state"] = {
                    "phase": "crew",
                    "message": (
                        "CrewAI is executing sequentially: "
                        "Assistant → Web Search Assistant → Entry Agent."
                    ),
                }
                render_buildathon_progress(
                    buildathon_progress_slot,
                    "crew",
                    "CrewAI is executing sequentially: "
                    "Assistant → Web Search Assistant → Entry Agent.",
                )

                with st.spinner("Running required three-agent CrewAI workflow..."):
                    core_result = run_agentic(
                        safe_query,
                        session_memory=buildathon_memory_context,
                        search_memory=buildathon_search_context,
                        progress_callback=lambda phase, msg: (
                            st.session_state.__setitem__(
                                "buildathon_progress_state",
                                {"phase": phase, "message": msg},
                            ),
                            render_buildathon_progress(
                                buildathon_progress_slot,
                                phase,
                                msg,
                            ),
                        ),
                    )

                st.session_state["buildathon_result"] = core_result

                output_guardrail = core_result.get("output_guardrail")

                # Backward compatibility for Streamlit hot-reload/session cache:
                # older cached results may predate the output_guardrail field.
                # Reconstruct it deterministically from the existing outputs
                # instead of treating missing metadata as an automatic failure.
                if output_guardrail is None:
                    output_guardrail = run_output_guardrail(
                        query=safe_query,
                        assistant_answer=core_result.get("assistant_answer", ""),
                        web_answer=core_result.get("web_answer", ""),
                        final_answer=(
                            core_result.get("final_answer")
                            or core_result.get("answer", "")
                        ),
                        temporal_validation=core_result.get(
                            "temporal_validation"
                        ),
                    )
                    core_result["output_guardrail"] = output_guardrail
                    st.session_state["buildathon_result"] = core_result

                if not output_guardrail.get("passed", False):
                    st.session_state["buildathon_evaluation"] = {
                        "status": "SKIPPED",
                        "result": {},
                        "openai_calls": 0,
                        "serper_calls": 0,
                        "reason": "Output guardrail blocked generated output before evaluation.",
                    }
                    st.session_state["latest_evaluation"] = None
                    st.session_state["buildathon_elapsed"] = (
                        time.perf_counter() - total_start
                    )
                    st.session_state["buildathon_progress_state"] = {
                        "phase": "output_blocked",
                        "message": (
                            "Generated output was blocked before persistence and "
                            "evaluation. No evaluator API call was made."
                        ),
                    }
                    render_buildathon_progress(
                        buildathon_progress_slot,
                        "output_blocked",
                        "Generated output was blocked before persistence and "
                        "evaluation. No evaluator API call was made.",
                    )
                    st.session_state["buildathon_notice"] = (
                        "The workflow generated output, but the Output Guardrail "
                        "blocked release/persistence. Review the guardrail evidence."
                    )
                    st.rerun()

                # Buildathon Core Step 3: evaluate the approved completed output.
                # IMPORTANT: reuse the existing Web Search Assistant output.
                # No second Serper/search request is made.
                st.session_state["buildathon_progress_state"] = {
                    "phase": "evaluation",
                    "message": (
                        "Output Guardrail passed. Evaluating the approved final "
                        "answer using existing workflow evidence — no second search."
                    ),
                }
                render_buildathon_progress(
                    buildathon_progress_slot,
                    "evaluation",
                    "Output Guardrail passed. Evaluating the approved final "
                    "answer using existing workflow evidence — no second search.",
                )

                with st.spinner("Evaluating completed workflow output..."):
                    buildathon_evaluation = evaluate_buildathon_from_existing_output(
                        safe_query,
                        core_result,
                    )

                st.session_state["buildathon_evaluation"] = buildathon_evaluation
                st.session_state["latest_evaluation"] = buildathon_evaluation
                st.session_state["buildathon_elapsed"] = time.perf_counter() - total_start

                if st.session_state.get("buildathon_memory_enabled", True):
                    add_session_memory_turn(
                        buildathon_query.strip(),
                        core_result.get("final_answer")
                        or core_result.get("answer", ""),
                        scope="buildathon",
                    )

                _cache_store(
                    "buildathon_run_cache",
                    buildathon_cache_identity,
                    {
                        "result": core_result,
                        "evaluation": buildathon_evaluation,
                        "security": security,
                        "elapsed": st.session_state["buildathon_elapsed"],
                    },
                )
                st.session_state["buildathon_progress_state"] = {
                    "phase": "complete",
                    "message": (
                        "Workflow complete — guardrails, three-agent execution, "
                        "persistence and evaluation finished successfully."
                    ),
                }
                render_buildathon_progress(
                    buildathon_progress_slot,
                    "complete",
                    "Workflow complete — guardrails, three-agent execution, "
                    "persistence and evaluation finished successfully.",
                )

                st.session_state["buildathon_notice"] = (
                    "Buildathon workflow and evaluation completed successfully."
                )
                st.rerun()

    buildathon_notice = st.session_state.pop("buildathon_notice", None)
    if buildathon_notice:
        st.success(buildathon_notice)

    buildathon_block_notice = st.session_state.pop(
        "buildathon_block_notice",
        None,
    )
    if buildathon_block_notice:
        st.error(buildathon_block_notice)

    core_result = st.session_state.get("buildathon_result")

    if core_result:
        st.divider()
        st.header("Results")

        left, right = st.columns(2)

        with left:
            st.subheader("Assistant Agent")
            st.write(core_result["assistant_answer"])

        with right:
            st.subheader("Web Search Assistant")
            st.write(core_result["web_answer"])

        st.subheader("Entry Agent · Final Grounded Answer")
        st.write(
            core_result.get("final_answer")
            or core_result.get("answer", "")
        )

        if core_result.get("saved_file"):
            st.success(
                f"Final grounded answer persisted to "
                f"{core_result['saved_file']}"
            )

        with st.expander("View persisted record"):
            st.text(core_result["entry_record"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Agents", core_result.get("agents", 3))
        m2.metric("Process", core_result.get("process", "Sequential"))
        m3.metric(
            "Persistence",
            "PASS" if core_result.get("persistence") else "FAILED"
        )
        m4.metric(
            "Execution Time",
            f"{core_result.get('elapsed_time', 0):.2f}s"
        )

        # ---------------- Detailed Security & Guardrails ----------------
        security = st.session_state.get("buildathon_security") or {}

        st.markdown("### 🛡 Security & Guardrails")
        st.caption(
            "Pre-execution controls · deterministic checks run before the "
            "three-agent workflow"
        )

        deterministic = security.get("deterministic") or {}
        for check in deterministic.get("checks", []):
            if check.get("status") == "PASS":
                st.success(
                    f"✅ {check.get('name')} — {check.get('detail', '')}"
                )
            else:
                st.error(
                    f"🛑 {check.get('name')} — {check.get('detail', '')}"
                )

        safety = security.get("ai_safety")
        if safety:
            if safety.get("allowed"):
                st.success(
                    f"✅ AI Safety — {safety.get('decision')} / "
                    f"{safety.get('category')} — {safety.get('reason')}"
                )
            else:
                st.error(
                    f"🛑 AI Safety — {safety.get('decision')} / "
                    f"{safety.get('category')} — {safety.get('reason')}"
                )

        st.markdown("#### Output Guardrail")
        st.caption(
            "Post-generation release gate · deterministic checks · zero "
            "additional OpenAI or search calls"
        )

        output_guardrail = core_result.get("output_guardrail") or {}
        if output_guardrail.get("passed"):
            st.success("✅ Output Guardrail — PASS")
        else:
            st.error("🛑 Output Guardrail — BLOCKED")

        output_checks = output_guardrail.get("checks", [])
        if output_checks:
            output_cols = st.columns(min(5, len(output_checks)))
            for col, check in zip(output_cols, output_checks):
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{check.get('name', 'Output check')}**")
                        if check.get("status") == "PASS":
                            st.success("PASS")
                        else:
                            st.error("FAIL")
                        st.caption(check.get("detail", ""))

        # ---------------- Workflow Evidence ----------------
        workflow_search = core_result.get("workflow_search") or {}
        workflow_items = workflow_search.get("items", [])

        st.markdown("### Evidence Used")
        st.caption(
            "The Web Search Assistant and Buildathon evaluator reuse this same "
            "workflow evidence. No second search is performed for Buildathon evaluation."
        )

        ev1, ev2, ev3 = st.columns(3)
        ev1.metric("Evidence Sources", len(workflow_items))
        ev2.metric("New Search Calls", workflow_search.get("serper_calls", 0))
        ev3.metric(
            "Session Reuse",
            "YES" if workflow_search.get("cache_hit") else "NO",
        )

        if workflow_search.get("cache_hit"):
            st.info(
                "Search evidence was reused from this session "
                f"({workflow_search.get('cache_match', 'cached match')}); "
                "no new workflow-search API call was required."
            )

        with st.expander("View workflow evidence", expanded=False):
            if workflow_search.get("search_query"):
                st.caption(
                    "Search query used: "
                    + str(workflow_search.get("search_query"))
                )

            if workflow_items:
                for index, item in enumerate(workflow_items, start=1):
                    st.markdown(
                        f"**Source {index}: {item.get('title') or 'Untitled source'}**"
                    )
                    if item.get("snippet"):
                        st.write(item.get("snippet"))
                    if item.get("link"):
                        st.markdown(f"Source link: {item.get('link')}")
                    if index < len(workflow_items):
                        st.divider()
            else:
                st.warning("No workflow evidence was available for this run.")

        st.info(
            "Why Buildathon grounding can be higher than Experiment 1: "
            "Buildathon **Workflow Grounding** checks the final answer against "
            "the same evidence used to generate it. Experiment 1 **Independent "
            "Groundedness** checks the answer against a separately retrieved "
            "evaluation evidence set, so the two scores are intentionally not "
            "directly comparable."
        )

        # ---------------- Buildathon Validation Evidence ----------------
        buildathon_evaluation = st.session_state.get("buildathon_evaluation")
        buildathon_validation = build_buildathon_validation(
            core_result,
            security,
            buildathon_evaluation,
        )

        st.markdown("### Validation Evidence")
        st.caption(
            "Measured execution checks · deterministic validation adds zero "
            "OpenAI or search calls"
        )

        status_fn = (
            st.success
            if buildathon_validation.get("status") == "PASS"
            else st.error
        )
        status_fn(
            f"{'✅' if buildathon_validation.get('status') == 'PASS' else '⚠️'} "
            f"Overall Buildathon Validation — "
            f"{buildathon_validation.get('status')}"
        )

        build_checks = buildathon_validation.get("checks", [])
        for row_start in range(0, len(build_checks), 3):
            row_checks = build_checks[row_start:row_start + 3]
            cols = st.columns(len(row_checks))
            for col, check in zip(cols, row_checks):
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{check['name']}**")
                        if check["status"] == "PASS":
                            st.success("PASS")
                        else:
                            st.error("FAIL")
                        st.caption(check["detail"])

        with st.expander("View Buildathon validation trace", expanded=False):
            for check in build_checks:
                st.write(
                    f"{'✅' if check['status'] == 'PASS' else '⚠️'} "
                    f"{check['name']} — {check['detail']}"
                )

        # Buildathon execution evidence only.
        # Deliberately no independent web search / evaluator here.
        st.markdown("### Buildathon Execution Evidence")
        st.caption(
            "Evidence shown below comes from the required Buildathon execution and its output evaluation. "
            "The evaluator reuses the workflow's existing raw search evidence; no duplicate evaluation search is performed."
        )

        deterministic_checks = deterministic.get("checks", [])

        passed_checks = sum(
            1 for check in deterministic_checks
            if check.get("status") == "PASS"
        )
        total_checks = len(deterministic_checks)

        ai_safety = security.get("ai_safety")
        if ai_safety:
            ai_safety_status = "PASS" if ai_safety.get("allowed") else "BLOCKED"
        else:
            ai_safety_status = "NOT CALLED"

        e1, e2, e3, e4 = st.columns(4)
        e1.metric(
            "Guardrails",
            "PASS" if security.get("allowed") else "BLOCKED"
        )
        e2.metric(
            "Deterministic Checks",
            f"{passed_checks}/{total_checks}" if total_checks else "N/A"
        )
        e3.metric("AI Safety", ai_safety_status)
        e4.metric(
            "Record Saved",
            "YES" if core_result.get("persistence") else "NO"
        )

        temporal = core_result.get("temporal_validation") or {}

        if temporal.get("applied"):
            if temporal.get("passed"):
                st.success(
                    "Temporal validation — PASS · "
                    + temporal.get("detail", "")
                )
            else:
                st.error(
                    "Temporal validation — FAIL · "
                    + temporal.get("detail", "")
                )

        st.info(
            "Buildathon Core performs one workflow search and reuses the same "
            "raw evidence for the Web Search Assistant and evaluation. "
            "Temporal validation is deterministic and makes no API call."
        )

elif navigation == "🧪  Experiments":
    st.subheader("Agent Usage Experiments")
    st.caption("Controlled proof-and-learning experiments. These are extensions to the buildathon core, not additional buildathon requirements.")
    experiment = st.radio(
        "Choose Learning Experiment",
        [
            "🧩 Experiment 1 · Fixed Sequential Workflow",
            "🧭 Experiment 2 · Dynamic Decision Workflow",
        ],
        horizontal=True,
    )
    if "Experiment 1" in experiment:
        render_experiment_overview("fixed")
        render_experiment_architecture("fixed")
        placeholder = "Example: What is the latest stable version of Python?"
    else:
        render_experiment_overview("dynamic")
        render_experiment_architecture("dynamic")
        placeholder = "Example: I was charged twice, I can't log in, and I need access urgently for a customer presentation."
    st.caption("Type the request and press Enter, or use the Run button.")
    with st.form("experiment_query_form", clear_on_submit=False):
        query = st.text_input(
            "Customer Request",
            placeholder=placeholder,
            key="experiment_query",
        )
        run_button = st.form_submit_button(
            "Run Experiment",
            type="primary",
            use_container_width=False,
        )

else:
    st.subheader("Learning Summary")
    st.caption("What the lab demonstrates about choosing agentic versus non-agentic architecture.")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### Use explicit workflows when")
        st.markdown("- The execution path is known in advance.\n- Rules are stable and deterministic.\n- Predictability, latency and cost are priorities.\n- Tool selection does not require semantic reasoning.")
    with c2:
        st.markdown("### Agents become useful when")
        st.markdown("- The required path varies with the request.\n- The system must interpret intent before choosing capabilities.\n- Tool combinations grow and hard-coded routing becomes difficult to maintain.\n- Runtime reasoning and orchestration provide useful flexibility.")
    st.markdown("### Core conclusion")
    st.info("Agents do not provide a capability that ordinary code can never implement. Their value is an orchestration abstraction: they can move some decision ownership from developer-written routing rules to runtime reasoning and tool selection.")
    st.markdown("### What the experiments prove")
    summary_df = pd.DataFrame({
        "Experiment": ["1 · Fixed workflow", "2 · Dynamic workflow"],
        "Decision model": ["Predetermined sequence", "Request-dependent capability selection"],
        "Non-agentic approach": ["Explicit Python orchestration", "Developer-written routing rules"],
        "Agentic approach": ["Sequential CrewAI agents", "Runtime tool selection"],
        "Primary lesson": ["Agents may add overhead when the path is already known", "Agents can reduce explicit routing logic as capability combinations grow"],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

if navigation == "🧪  Experiments" and run_button:
    experiment_cache_name = (
        "experiment1_run_cache"
        if "Experiment 1" in experiment
        else "experiment2_run_cache"
    )
    cached_experiment = None
    if query.strip():
        cached_experiment, _ = _cache_lookup(
            experiment_cache_name,
            query.strip(),
            allow_near=False,
        )

    if cached_experiment is not None:
        clear_run_state()
        for state_key, state_value in cached_experiment.get("state", {}).items():
            st.session_state[state_key] = state_value
        st.session_state["experiment_notice"] = (
            "Reused the exact experiment result from this session — no new "
            "OpenAI or search API calls were required. Metrics shown are from "
            "the original measured run."
        )
        st.rerun()

    clear_run_state()
    st.session_state["experiment"] = experiment
    total_start = time.perf_counter()

    with st.spinner("Running common security gate..."):
        security = run_security_gate(query)
    st.session_state["security"] = security

    if not security["allowed"]:
        st.session_state["experiment_block_notice"] = (
            "Request blocked by the security gate. No workflow, tool, search "
            "or evaluation stage was executed."
        )
        st.rerun()
    else:
        safe_query = security["deterministic"]["query"]

        if "Experiment 1" in experiment:
            # Fair comparison + lower search cost: both architectures receive
            # the SAME workflow evidence retrieved once.
            with st.spinner("Retrieving one shared workflow evidence set..."):
                shared_workflow_search = run_non_agentic_web_search(safe_query)

            with st.spinner("Running explicit Python orchestration..."):
                non_agentic = run_non_agentic(
                    safe_query,
                    workflow_search=shared_workflow_search,
                )

            with st.spinner("Running required three-agent CrewAI workflow..."):
                agentic = run_agentic(
                    safe_query,
                    workflow_search=shared_workflow_search,
                )

            # Keep evaluation independent from generation evidence. This is the
            # only additional search used for Experiment 1 validation.
            with st.spinner("Retrieving one independent evaluation evidence set..."):
                evidence = retrieve_evaluation_evidence(safe_query)

            with st.spinner("Running one consolidated evaluator..."):
                evaluation = run_common_evaluation(
                    safe_query,
                    evidence,
                    non_agentic,
                    agentic,
                )

            validation = build_experiment1_validation(
                non_agentic,
                agentic,
                evaluation,
            )

            st.session_state["non_agentic"] = non_agentic
            st.session_state["agentic"] = agentic
            st.session_state["shared_workflow_search"] = shared_workflow_search
            st.session_state["evidence"] = evidence
            st.session_state["evaluation"] = evaluation
            st.session_state["latest_evaluation"] = evaluation
            st.session_state["experiment1_validation"] = validation
            st.session_state["observations"] = build_observations(non_agentic, agentic, evaluation)
            st.session_state["run_metrics"] = {
                "total_elapsed_time": time.perf_counter()-total_start,
                "known_openai_calls": security.get("openai_calls",0)+non_agentic.get("known_openai_calls",0)+evaluation.get("openai_calls",0),
                "crew_model_calls": "Framework-controlled",
                "workflow_serper_calls": shared_workflow_search.get("serper_calls",0),
                "evaluation_serper_calls": evidence.get("serper_calls",0),
                "cache_hits": (
                    security.get("cache_hits", 0)
                    + (1 if shared_workflow_search.get("cache_hit") else 0)
                    + (1 if evidence.get("cache_hit") else 0)
                ),
                "workflow_cache_match": shared_workflow_search.get("cache_match"),
                "evaluation_cache_match": evidence.get("cache_match"),
            }
            _cache_store(
                "experiment1_run_cache",
                safe_query,
                {
                    "state": {
                        "experiment": experiment,
                        "security": security,
                        "non_agentic": non_agentic,
                        "agentic": agentic,
                        "shared_workflow_search": shared_workflow_search,
                        "evidence": evidence,
                        "evaluation": evaluation,
                        "latest_evaluation": evaluation,
                        "experiment1_validation": validation,
                        "observations": st.session_state["observations"],
                        "run_metrics": st.session_state["run_metrics"],
                    }
                },
            )
        else:
            with st.spinner("Running explicit Python decision rules..."):
                dynamic_non = run_dynamic_non_agentic(safe_query)
            with st.spinner("Running dynamic CrewAI tool-selection agent..."):
                dynamic_agent = run_dynamic_agentic(safe_query)

            action_validation = validate_agent_actions(dynamic_agent)
            decision_validation = build_dynamic_decision_validation(
                dynamic_non,
                dynamic_agent,
                action_validation,
            )

            with st.spinner("Running one consolidated Experiment 2 evaluator..."):
                dynamic_evaluation = run_dynamic_response_evaluation(
                    safe_query,
                    dynamic_non,
                    dynamic_agent,
                )

            st.session_state["dynamic_non_agentic"] = dynamic_non
            st.session_state["dynamic_agentic"] = dynamic_agent
            st.session_state["dynamic_action_validation"] = action_validation
            st.session_state["dynamic_decision_validation"] = decision_validation
            st.session_state["dynamic_evaluation"] = dynamic_evaluation
            st.session_state["latest_evaluation"] = dynamic_evaluation
            st.session_state["dynamic_observations"] = build_dynamic_observations(
                dynamic_non,
                dynamic_agent,
                action_validation,
            )
            st.session_state["run_metrics"] = {
                "total_elapsed_time": time.perf_counter()-total_start,
                "known_openai_calls": (
                    security.get("openai_calls", 0)
                    + dynamic_evaluation.get("openai_calls", 0)
                ),
                "crew_model_calls": "Framework-controlled",
                "workflow_serper_calls": 0,
                "evaluation_serper_calls": 0,
                "cache_hits": security.get("cache_hits", 0),
                "workflow_cache_match": None,
                "evaluation_cache_match": None,
            }
            _cache_store(
                "experiment2_run_cache",
                safe_query,
                {
                    "state": {
                        "experiment": experiment,
                        "security": security,
                        "dynamic_non_agentic": dynamic_non,
                        "dynamic_agentic": dynamic_agent,
                        "dynamic_action_validation": action_validation,
                        "dynamic_decision_validation": decision_validation,
                        "dynamic_evaluation": dynamic_evaluation,
                        "latest_evaluation": dynamic_evaluation,
                        "dynamic_observations": st.session_state["dynamic_observations"],
                        "run_metrics": st.session_state["run_metrics"],
                    }
                },
            )
        st.session_state["experiment_notice"] = "Experiment completed successfully."
        st.rerun()

# Show the completion notice after rerun so sidebar evidence is already current.
if navigation == "🧪  Experiments":
    experiment_notice = st.session_state.pop("experiment_notice", None)
    if experiment_notice:
        st.success(experiment_notice)

    experiment_block_notice = st.session_state.pop(
        "experiment_block_notice",
        None,
    )
    if experiment_block_notice:
        st.error(experiment_block_notice)

# Common security panel
security = st.session_state.get("security")
if navigation == "🧪  Experiments" and security:
    st.divider(); st.header("🛡 Security & Guardrails")
    for check in security["deterministic"]["checks"]:
        (st.success if check["status"]=="PASS" else st.error)(f"{'✅' if check['status']=='PASS' else '🛑'} {check['name']} — {check['detail']}")
    safety = security.get("ai_safety")
    if safety:
        (st.success if safety["allowed"] else st.error)(f"{'✅' if safety['allowed'] else '🛑'} AI Safety — {safety['decision']} / {safety['category']} — {safety['reason']}")

# Experiment 1 panels
non_agentic = st.session_state.get("non_agentic")
agentic = st.session_state.get("agentic")
if navigation == "🧪  Experiments" and non_agentic and agentic:
    st.divider()
    st.header("Experiment 1 · Fixed Sequential Workflow")
    st.caption(
        "Controlled comparison: the same predictable support task and the same "
        "workflow evidence are processed through explicit Python orchestration "
        "and the required three-agent CrewAI sequence."
    )

    # ---------------- Side-by-side execution results ----------------
    st.markdown("#### Execution Results · Side-by-Side")
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Non-Agentic · Python Orchestration")
        st.markdown("**Initial model answer**")
        st.write(non_agentic["assistant_answer"])
        st.markdown("**Final web-grounded answer**")
        st.write(non_agentic["web_answer"])
        with st.expander("View persisted Python record"):
            st.text(non_agentic["entry_record"])

    with right:
        st.subheader("Agentic · CrewAI")
        st.markdown("**Assistant Agent**")
        st.write(agentic["assistant_answer"])
        st.markdown("**Web Search Assistant**")
        st.write(agentic["web_answer"])
        st.markdown("**Entry Agent · Final Grounded Answer**")
        st.write(agentic.get("final_answer") or agentic.get("answer", ""))
        with st.expander("View persisted CrewAI record"):
            st.text(agentic["entry_record"])

    # ---------------- Architecture / execution metrics ----------------
    st.markdown("#### Architecture & Execution Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Python Agents", 0)
    m2.metric("CrewAI Agents", agentic.get("agents", 3))
    m3.metric("Python Time", f"{non_agentic['elapsed_time']:.2f}s")
    m4.metric("CrewAI Time", f"{agentic['elapsed_time']:.2f}s")
    m5.metric("Workflow Searches", st.session_state.get("run_metrics", {}).get("workflow_serper_calls", 0))
    m6.metric("Eval Searches", st.session_state.get("run_metrics", {}).get("evaluation_serper_calls", 0))

    # ---------------- Clear comparison snapshot ----------------
    evaluation = st.session_state.get("evaluation")
    if evaluation and evaluation.get("status") == "SUCCESS":
        result = evaluation["result"]

        st.markdown("#### Comparison Snapshot")
        st.caption(
            "The same request and shared workflow evidence, compared as two "
            "different orchestration styles."
        )

        python_time = non_agentic["elapsed_time"]
        crew_time = agentic["elapsed_time"]
        latency_delta = crew_time - python_time

        metrics_order = [
            "relevance",
            "completeness",
            "consistency",
            "groundedness",
        ]
        python_scores = [
            result["non_agentic"][metric]["score"]
            for metric in metrics_order
        ]
        crew_scores = [
            result["agentic"][metric]["score"]
            for metric in metrics_order
        ]
        python_avg = round(sum(python_scores) / len(python_scores))
        crew_avg = round(sum(crew_scores) / len(crew_scores))
        max_quality_delta = max(
            abs(a - b) for a, b in zip(python_scores, crew_scores)
        )

        left_snapshot, right_snapshot = st.columns(2, gap="large")
        with left_snapshot:
            with st.container(border=True):
                st.markdown("### 🐍 Explicit Python")
                st.caption("Predetermined orchestration · 0 agents")
                p1, p2, p3 = st.columns(3)
                p1.metric("Execution", f"{python_time:.2f}s")
                p2.metric("Avg Quality", f"{python_avg}/100")
                p3.metric(
                    "Independent Groundedness",
                    f"{result['non_agentic']['groundedness']['score']}/100",
                    help=(
                        "Support against a separate independent evaluation "
                        "evidence set retrieved after generation."
                    ),
                )

        with right_snapshot:
            with st.container(border=True):
                st.markdown("### 🤖 CrewAI")
                st.caption("Role-based orchestration · 3 agents")
                a1, a2, a3 = st.columns(3)
                a1.metric("Execution", f"{crew_time:.2f}s")
                a2.metric("Avg Quality", f"{crew_avg}/100")
                a3.metric(
                    "Independent Groundedness",
                    f"{result['agentic']['groundedness']['score']}/100",
                    help=(
                        "Support against a separate independent evaluation "
                        "evidence set retrieved after generation."
                    ),
                )

        takeaway_left, takeaway_right = st.columns(2)
        with takeaway_left:
            st.info(
                f"**Quality spread:** the largest score difference between "
                f"the two approaches is **{max_quality_delta} points**."
            )
        with takeaway_right:
            if latency_delta >= 0:
                st.info(
                    f"**Orchestration overhead:** CrewAI took "
                    f"**{latency_delta:.2f}s longer** in this measured run."
                )
            else:
                st.info(
                    f"**Execution difference:** CrewAI completed "
                    f"**{abs(latency_delta):.2f}s faster** in this measured run."
                )

        search_metrics = st.session_state.get("run_metrics", {})
        st.caption(
            "Search usage · workflow: "
            f"{search_metrics.get('workflow_serper_calls', 0)} new call(s) · "
            "independent evaluation: "
            f"{search_metrics.get('evaluation_serper_calls', 0)} new call(s) · "
            f"cache hits: {search_metrics.get('cache_hits', 0)}."
        )

        st.markdown("#### Common Evaluation · Independent Evidence")
        st.caption(
            "Both architectures are scored separately against the same "
            "independent evidence set retrieved after generation."
        )

        eval_left, eval_right = st.columns(2, gap="large")

        with eval_left:
            with st.container(border=True):
                st.markdown("### 🐍 Explicit Python Evaluation")
                st.caption("Independent evaluation · same evidence set")

                ep1, ep2 = st.columns(2)
                ep1.metric(
                    "Relevance",
                    f"{result['non_agentic']['relevance']['score']}/100",
                )
                ep2.metric(
                    "Completeness",
                    f"{result['non_agentic']['completeness']['score']}/100",
                )

                ep3, ep4 = st.columns(2)
                ep3.metric(
                    "Consistency",
                    f"{result['non_agentic']['consistency']['score']}/100",
                )
                ep4.metric(
                    "Independent Groundedness",
                    f"{result['non_agentic']['groundedness']['score']}/100",
                    help=(
                        "How well factual claims are supported by the separate "
                        "independent evaluation evidence, not by the workflow's "
                        "own search evidence."
                    ),
                )

        with eval_right:
            with st.container(border=True):
                st.markdown("### 🤖 CrewAI Evaluation")
                st.caption("Independent evaluation · same evidence set")

                ea1, ea2 = st.columns(2)
                ea1.metric(
                    "Relevance",
                    f"{result['agentic']['relevance']['score']}/100",
                )
                ea2.metric(
                    "Completeness",
                    f"{result['agentic']['completeness']['score']}/100",
                )

                ea3, ea4 = st.columns(2)
                ea3.metric(
                    "Consistency",
                    f"{result['agentic']['consistency']['score']}/100",
                )
                ea4.metric(
                    "Independent Groundedness",
                    f"{result['agentic']['groundedness']['score']}/100",
                    help=(
                        "How well factual claims are supported by the separate "
                        "independent evaluation evidence, not by the workflow's "
                        "own search evidence."
                    ),
                )

        comparison = result.get("comparison", {})
        if comparison.get("available"):
            st.info(
                f"Answer relationship: {comparison['agreement']} — "
                f"{comparison['reason']}"
            )

        non_ground = result["non_agentic"]["groundedness"]["score"]
        agent_ground = result["agentic"]["groundedness"]["score"]
        min_ground = min(non_ground, agent_ground)

        st.info(
            "**What does Independent Groundedness mean?**  "
            "It measures how well the generated factual claims are supported "
            "by a separate evaluation evidence set retrieved after generation. "
            "It is not a score against the other architecture, and it is not "
            "the same as Buildathon workflow grounding, which checks against "
            "the evidence already used to create the answer."
        )

        ground_reason_left = (
            result["non_agentic"]["groundedness"].get("reason", "")
        )
        ground_reason_right = (
            result["agentic"]["groundedness"].get("reason", "")
        )

        with st.expander(
            "Why did Independent Groundedness receive these scores?",
            expanded=False,
        ):
            st.markdown(
                f"**Explicit Python — {non_ground}/100**  \n"
                f"{ground_reason_left or 'No evaluator reason was returned.'}"
            )
            st.markdown(
                f"**CrewAI — {agent_ground}/100**  \n"
                f"{ground_reason_right or 'No evaluator reason was returned.'}"
            )

        if min_ground >= 85:
            st.success(
                "Strong independent evidence alignment — both architectures "
                f"scored at least {min_ground}/100."
            )
        elif min_ground >= 70:
            st.info(
                "Moderate independent evidence alignment — the lower "
                f"Independent Groundedness score is {min_ground}/100."
            )
        else:
            st.warning(
                "Low independent evidence alignment — at least one architecture "
                f"scored below 70/100 (lower score: {min_ground}/100). "
                "The answers may agree with each other while still being only "
                "partially supported by the independent evidence."
            )

    # ---------------- Deterministic validation evidence ----------------
    validation = st.session_state.get("experiment1_validation")
    if validation:
        st.markdown("#### Validation Evidence")
        st.caption(
            "Measured execution checks · deterministic validation adds zero "
            "OpenAI or search calls"
        )

        status_fn = st.success if validation.get("status") == "PASS" else st.error
        status_fn(
            f"{'✅' if validation.get('status') == 'PASS' else '⚠️'} "
            f"Overall Experiment Validation — {validation.get('status')}"
        )

        checks = validation.get("checks", [])
        for row_start in range(0, len(checks), 3):
            row_checks = checks[row_start:row_start + 3]
            cols = st.columns(len(row_checks))
            for col, check in zip(cols, row_checks):
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{check['name']}**")
                        if check["status"] == "PASS":
                            st.success("PASS")
                        else:
                            st.error("FAIL")
                        st.caption(check["detail"])

        with st.expander("View validation trace"):
            for check in checks:
                st.write(
                    f"{'✅' if check['status'] == 'PASS' else '⚠️'} "
                    f"{check['name']} — {check['detail']}"
                )

    # ---------------- Observations ----------------
    with st.expander("View Detailed Observations", expanded=False):
        for item in st.session_state.get("observations", []):
            with st.container(border=True):
                st.markdown(f"**{item['title']}**")
                st.write(item["text"])

    st.info(
        "Learning takeaway: both architectures can solve this known sequential "
        "workflow. Experiment 1 measures whether role-based agent orchestration "
        "adds enough value to justify its additional orchestration overhead when "
        "the execution path is already predetermined."
    )

# Experiment 2 panels
dyn_non = st.session_state.get("dynamic_non_agentic")
dyn_agent = st.session_state.get("dynamic_agentic")

if navigation == "🧪  Experiments" and dyn_non and dyn_agent:
    st.divider()
    st.header("Experiment 2 · Dynamic Decision Workflow")
    st.caption(
        "The same four simulated enterprise capabilities are available to both "
        "approaches. Python evaluates developer-written rules; the agent selects "
        "useful capabilities at runtime. Simulated tools make zero external API calls."
    )

    st.markdown("#### Decision Path · Side-by-Side")
    path_left, path_right = st.columns(2, gap="large")

    with path_left:
        st.markdown(
            """
            <div class='compare-path non'>
              <div class='compare-kicker'>Non-Agentic</div>
              <div class='compare-title'>Developer-Written Routing</div>
              <div class='compare-step'>1 · Receive customer request</div>
              <div class='compare-step'>2 · Evaluate predefined Python rules</div>
              <div class='compare-step'>3 · Invoke every matching capability</div>
              <div class='compare-step'>4 · Execute deterministic actions</div>
              <div class='compare-foot'><b>Decision owner:</b> developer-authored
              rules written before runtime.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with path_right:
        st.markdown(
            """
            <div class='compare-path agent'>
              <div class='compare-kicker'>Agentic</div>
              <div class='compare-title'>Runtime Capability Selection</div>
              <div class='compare-step'>1 · Receive customer request</div>
              <div class='compare-step'>2 · Interpret intent and context</div>
              <div class='compare-step'>3 · Select relevant tools at runtime</div>
              <div class='compare-step'>4 · Execute selected tools and respond</div>
              <div class='compare-foot'><b>Decision owner:</b> runtime agent
              reasoning within the available tool boundaries.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "Experiment 2 tests decision ownership, not an impossible-without-agents "
        "capability. Explicit Python can implement the same actions, but the "
        "developer must encode and maintain the routing logic."
    )

    st.markdown("#### Execution Results · Side-by-Side")
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Non-Agentic · Explicit Routing")
        st.write(dyn_non["answer"])
        with st.expander("View Python rule trace", expanded=False):
            for branch in dyn_non["branches"]:
                st.write(
                    f"{'✅' if branch['matched'] else '○'} "
                    f"{branch['rule']} → {branch['tool']}"
                )

    with right:
        st.subheader("Agentic · Runtime Tool Selection")
        st.write(dyn_agent["answer"])
        with st.expander("View runtime tool trace", expanded=False):
            if dyn_agent["tool_trace"]:
                for step, item in enumerate(dyn_agent["tool_trace"], 1):
                    st.write(f"{step}. {item['tool']} → {item['result']}")
            else:
                st.write("No simulated support tool was invoked.")

    decision_validation = st.session_state.get("dynamic_decision_validation") or {}
    agreement = decision_validation.get("selection_agreement", 0)
    capability_count = decision_validation.get("capability_count", 4)
    latency_overhead = dyn_agent["elapsed_time"] - dyn_non["elapsed_time"]

    st.markdown("#### Decision Comparison Snapshot")
    st.caption(
        "A direct view of routing effort, capability selection and measured overhead."
    )

    snap_left, snap_right = st.columns(2, gap="large")

    with snap_left:
        with st.container(border=True):
            st.markdown("### 🐍 Explicit Python")
            st.caption("Developer-owned routing · fixed rules")
            p1, p2, p3 = st.columns(3)
            p1.metric("Rules Evaluated", dyn_non["routing_rules"])
            p2.metric("Tools Used", len(set(dyn_non["tools_used"])))
            p3.metric("Execution", f"{dyn_non['elapsed_time']:.2f}s")

    with snap_right:
        with st.container(border=True):
            st.markdown("### 🤖 Runtime Agent")
            st.caption("Context-aware runtime capability selection")
            a1, a2, a3 = st.columns(3)
            a1.metric("Tools Used", len(dyn_agent["tools_used"]))
            a2.metric("Tools Avoided", dyn_agent["tools_avoided"])
            a3.metric("Execution", f"{dyn_agent['elapsed_time']:.2f}s")

    compare_left, compare_right = st.columns(2)

    with compare_left:
        st.info(
            f"**Capability-selection agreement:** {agreement}/{capability_count} "
            "capabilities had the same USED/AVOIDED decision in this run. "
            "This is comparison evidence, not a claim that Python rules are "
            "absolute ground truth."
        )

    with compare_right:
        st.info(
            f"**Runtime orchestration overhead:** the agentic path took "
            f"{latency_overhead:+.2f}s relative to explicit Python."
        )

    st.markdown("#### Capability Selection")
    st.caption("USED = capability executed · AVOIDED = capability not invoked")

    selection_rows = decision_validation.get("selection_rows", [])

    if selection_rows:
        cols = st.columns(len(selection_rows))
        for col, row in zip(cols, selection_rows):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{row['capability']}**")
                    st.write(
                        "Python: "
                        + ("**USED**" if row["python_selected"] else "AVOIDED")
                    )
                    st.write(
                        "Agent: "
                        + ("**USED**" if row["agent_selected"] else "AVOIDED")
                    )

                    if row["same"]:
                        st.success("Same decision")
                    else:
                        st.info("Different decision")

    dynamic_evaluation = st.session_state.get("dynamic_evaluation")

    if dynamic_evaluation and dynamic_evaluation.get("status") == "SUCCESS":
        eval_result = dynamic_evaluation["result"]

        st.markdown("#### Response Evaluation · Tool-Trace Grounded")
        st.caption(
            "One consolidated evaluator scores both responses against the customer "
            "request and their actual simulated execution traces · zero web searches"
        )

        eval_left, eval_right = st.columns(2, gap="large")

        with eval_left:
            with st.container(border=True):
                st.markdown("### 🐍 Explicit Python Evaluation")
                st.caption("Response quality · grounded against Python action trace")

                ep1, ep2 = st.columns(2)
                ep1.metric(
                    "Relevance",
                    f"{eval_result['non_agentic']['relevance']['score']}/100",
                )
                ep2.metric(
                    "Completeness",
                    f"{eval_result['non_agentic']['completeness']['score']}/100",
                )

                ep3, ep4 = st.columns(2)
                ep3.metric(
                    "Consistency",
                    f"{eval_result['non_agentic']['consistency']['score']}/100",
                )
                ep4.metric(
                    "Trace Grounding",
                    f"{eval_result['non_agentic']['groundedness']['score']}/100",
                    help=(
                        "How well operational claims in the response are supported "
                        "by the actual simulated Python execution trace."
                    ),
                )

        with eval_right:
            with st.container(border=True):
                st.markdown("### 🤖 Runtime Agent Evaluation")
                st.caption("Response quality · grounded against actual tool trace")

                ea1, ea2 = st.columns(2)
                ea1.metric(
                    "Relevance",
                    f"{eval_result['agentic']['relevance']['score']}/100",
                )
                ea2.metric(
                    "Completeness",
                    f"{eval_result['agentic']['completeness']['score']}/100",
                )

                ea3, ea4 = st.columns(2)
                ea3.metric(
                    "Consistency",
                    f"{eval_result['agentic']['consistency']['score']}/100",
                )
                ea4.metric(
                    "Trace Grounding",
                    f"{eval_result['agentic']['groundedness']['score']}/100",
                    help=(
                        "How well operational claims in the response are supported "
                        "by the actual simulated runtime tool trace."
                    ),
                )

        comparison = eval_result.get("comparison") or {}
        if comparison.get("available"):
            st.info(
                f"**Response relationship:** {comparison.get('agreement')} — "
                f"{comparison.get('reason', '')}"
            )

        st.info(
            "**Trace Grounding** is different from Experiment 1's Independent "
            "Groundedness. Here, the score asks whether claimed support actions "
            "are backed by the actual simulated execution trace. No web evidence "
            "or additional search is required."
        )

        with st.expander("Why did the evaluator give these scores?", expanded=False):
            st.markdown(
                f"**Explicit Python · Trace Grounding — "
                f"{eval_result['non_agentic']['groundedness']['score']}/100**  \n"
                f"{eval_result['non_agentic']['groundedness'].get('reason', '')}"
            )
            st.markdown(
                f"**Runtime Agent · Trace Grounding — "
                f"{eval_result['agentic']['groundedness']['score']}/100**  \n"
                f"{eval_result['agentic']['groundedness'].get('reason', '')}"
            )

    if decision_validation:
        st.markdown("#### Decision Validation Evidence")
        st.caption(
            "Measured routing/tool-trace checks · deterministic validation adds "
            "zero OpenAI or search calls"
        )

        if decision_validation.get("status") == "PASS":
            st.success("✅ Overall Decision Validation — PASS")
        else:
            st.error("⚠️ Overall Decision Validation — FAIL")

        checks = decision_validation.get("checks", [])

        for row_start in range(0, len(checks), 3):
            row_checks = checks[row_start:row_start + 3]
            cols = st.columns(len(row_checks))

            for col, check in zip(cols, row_checks):
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{check['name']}**")
                        if check["status"] == "PASS":
                            st.success("PASS")
                        else:
                            st.error("FAIL")
                        st.caption(check["detail"])

        with st.expander("View decision-validation trace", expanded=False):
            for check in checks:
                st.write(
                    f"{'✅' if check['status'] == 'PASS' else '⚠️'} "
                    f"{check['name']} — {check['detail']}"
                )

    with st.expander("View Detailed Observations", expanded=False):
        for item in st.session_state.get("dynamic_observations", []):
            with st.container(border=True):
                st.markdown(f"**{item['title']}**")
                st.write(item["text"])

    st.info(
        "Learning takeaway: dynamic workflows can be implemented with explicit "
        "Python rules, but each routing condition and combination must be encoded "
        "and maintained by the developer. An agent can interpret the request and "
        "select from bounded capabilities at runtime, trading additional model/"
        "orchestration overhead for more flexible decision ownership."
    )

# API efficiency / transparency
metrics=st.session_state.get("run_metrics")
if navigation == "🧪  Experiments" and metrics:
    st.divider(); st.header("⚡ API Efficiency & Execution Transparency")
    c1,c2,c3,c4,c5,c6=st.columns(6)
    c1.metric("End-to-End",f"{metrics['total_elapsed_time']:.2f}s")
    c2.metric("Known Direct OpenAI",metrics["known_openai_calls"])
    c3.metric("CrewAI Model Calls",metrics["crew_model_calls"])
    c4.metric("Workflow Searches",metrics["workflow_serper_calls"])
    c5.metric("Eval Searches",metrics["evaluation_serper_calls"])
    c6.metric("Session Cache Hits",metrics.get("cache_hits",0))

    with st.expander("What these API-efficiency metrics mean", expanded=False):
        st.markdown(
            "- **End-to-End** — total measured time for this experiment run.\n"
            "- **Known Direct OpenAI** — model calls made explicitly by this app "
            "(for example safety classification, direct Python-path generation, "
            "and the common evaluator).\n"
            "- **CrewAI Model Calls** — CrewAI controls its internal model "
            "interactions, so the app deliberately does not invent an exact count.\n"
            "- **Workflow Searches** — external search calls used to ground the "
            "answer-generation workflow. A cache reuse counts as 0 new calls.\n"
            "- **Eval Searches** — separate independent web search is used only "
            "for Experiment 1. Experiment 2 uses one model evaluator against the "
            "existing simulated tool traces, so its Eval Searches remain 0.\n"
            "- **Session Cache Hits** — exact safety results and exact/very-near "
            "search evidence reused during this browser session instead of making "
            "another external API call."
        )

        workflow_match = metrics.get("workflow_cache_match")
        eval_match = metrics.get("evaluation_cache_match")
        if workflow_match or eval_match:
            st.caption(
                f"Cache reuse this run · workflow: {workflow_match or 'none'} · "
                f"evaluation: {eval_match or 'none'}"
            )

    st.success(
        "Avoided unnecessary calls: deterministic guardrails, validation, "
        "persistence, metrics, observations, architecture rendering and UI use "
        "no LLM calls. Repeated evidence requests can reuse the session cache, "
        "and blocked Layer-1 requests stop before any OpenAI call."
    )


    

