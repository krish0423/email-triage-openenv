"""
LLM-as-judge grader for Task 3 draft replies.
Falls back to heuristic scoring if LLM call fails or is not configured.
"""
import os
import json
from typing import Tuple

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False


JUDGE_SYSTEM_PROMPT = """You are an expert customer support quality evaluator.
You will be given a customer email and a draft reply written by an AI agent.
Score the draft reply from 0.0 to 1.0 based on these criteria:

1. Empathy & Tone (0.0-0.25): Does the reply acknowledge the customer's situation with warmth?
2. Accuracy & Relevance (0.0-0.25): Does it correctly address the specific issue?
3. Clear Next Step (0.0-0.25): Does it tell the customer what will happen next?
4. Professionalism (0.0-0.25): Is it professional, concise, and free of errors?

Return ONLY a JSON object with this exact format (no markdown, no preamble):
{\"score\": 0.75, \"reasoning\": \"brief one-sentence explanation\"}
"""

JUDGE_USER_TEMPLATE = """Customer Email:
Subject: {subject}
Body: {body}

AI Agent Draft Reply:
{draft_reply}

Score this reply (0.0 to 1.0):"""


def llm_judge_score(
    subject: str,
    body: str,
    draft_reply: str,
) -> Tuple[float, str]:
    """
    Use an LLM to score a draft reply for Task 3.
    Returns (score 0.0-1.0, reasoning string).
    Falls back to heuristic if LLM unavailable.
    """
    # Always return a tuple (score, reason)
    if not _OPENAI_AVAILABLE:
        return _heuristic_score(draft_reply)

    api_key = os.environ.get("HF_TOKEN", "")
    base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    if not api_key:
        return _heuristic_score(draft_reply)

    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                    subject=subject, body=body, draft_reply=draft_reply
                )},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        raw = getattr(resp.choices[0].message, "content", "") or ""
        raw = raw.strip()

        # Robustly strip fenced code blocks and leading text
        if raw.startswith("```"):
            # remove first fence and trailing fence if present
            parts = raw.split("```")
            # find the first part that looks like JSON
            candidate = ""
            for p in parts:
                p = p.strip()
                if p.startswith("{") and p.endswith("}"):
                    candidate = p
                    break
                # handle ```json\n{...}
                if p.lower().startswith("json") and "{" in p:
                    candidate = p[p.find("{"):]
                    break
            if candidate:
                raw = candidate
            else:
                # fallback to the middle segment if nothing obvious
                raw = parts[1] if len(parts) > 1 else raw

        # Try to extract JSON object from text
        start = raw.find("{")
        end = raw.rfind("}") + 1
        json_text = raw[start:end] if start != -1 and end != -1 and end > start else raw

        data = json.loads(json_text.strip())
        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))   # clamp
        reason = data.get("reasoning", "LLM judged")
        return score, reason

    except Exception as e:
        # Graceful fallback – never crash the environment
        score, reason = _heuristic_score(draft_reply)
        return score, f"LLM judge failed ({e}), heuristic used: {reason}"


def _heuristic_score(draft: str) -> Tuple[float, str]:
    """
    Rule-based fallback scorer.
    Checks length, empathy words, and action words.
    Max 1.0.
    """
    if not draft or len(draft.strip()) < 20:
        return 0.0, "Reply too short or missing."

    score = 0.0
    reasons = []

    # Length tiers (character-based but roughly correlates to words)
    length = len(draft)
    if length >= 50:
        score += 0.1
    if length >= 120:
        score += 0.1
    if length >= 250:
        score += 0.05
    reasons.append(f"length={length}")

    # Empathy signals
    empathy = ["understand", "apologize", "sorry", "appreciate", "thank", "concern", "frustrat"]
    hits = [w for w in empathy if w in draft.lower()]
    if hits:
        score += 0.25
        reasons.append(f"empathy={hits[0]}")

    # Action/resolution signals
    action_words = ["will", "team", "refund", "contact", "resolve", "fix", "escalate",
                    "investigate", "within", "hours", "days", "shortly", "immediately"]
    hits2 = [w for w in action_words if w in draft.lower()]
    if hits2:
        score += 0.25
        reasons.append(f"action={hits2[0]}")

    # Professionalism: greeting + closing
    has_greeting = any(g in draft.lower() for g in ["dear", "hello", "hi ", "thank you for"])
    has_closing = any(c in draft.lower() for c in ["regards", "sincerely", "best", "team"])
    if has_greeting:
        score += 0.15
    if has_closing:
        score += 0.1

    return round(min(score, 1.0), 4), ", ".join(reasons) or "heuristic scored"
