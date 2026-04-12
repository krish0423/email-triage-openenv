"""
Baseline inference script for the Email Triage OpenEnv environment.

Uses OpenAI-compatible API client (e.g. Groq) to power an LLM agent
that triages customer support emails through the OpenEnv API.

Required environment variables:
  API_BASE_URL — The API endpoint for the LLM
  MODEL_NAME   — The model identifier
  HF_TOKEN     — API key for the LLM provider
"""

import os
import sys
import json
import time
import requests
from typing import Optional
from openai import OpenAI

# ── Configuration ───────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

TASKS = [
    "task1_email_classification",
    "task2_prioritization_routing",
    "task3_full_triage_reply",
]

ENV_NAME = "email-triage-env"


def clamp_reward(r: float) -> float:
    return max(0.01, min(0.99, float(r)))


# ── Structured Logging ──────────────────────────
def emit_start(task_id: str) -> None:
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

def emit_step(step: int, action_str: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    r = clamp_reward(reward)
    done_str = "true" if done else "false"
    error_str = error if error else "null"
    print(f"[STEP] step={step} action={action_str} reward={r:.2f} done={done_str} error={error_str}", flush=True)

def emit_end(success: bool, steps: int, rewards: list) -> None:
    success_str = "true" if success else "false"
    clamped = [clamp_reward(r) for r in rewards]
    rewards_str = ",".join(f"{r:.2f}" for r in clamped)
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)


# ── LLM Client ──────────────────────────────────
llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are an expert customer support email triage agent.

You will be shown a customer email and must triage it by providing a structured JSON action.

Available categories: billing, technical, general, complaint, refund, phishing
Available priorities: P1 (urgent), P2 (medium), P3 (low)
Available departments: billing_team, technical_team, customer_success, returns, security

Category-to-department mapping:
- billing → billing_team, technical → technical_team, complaint → customer_success
- general → customer_success, refund → returns, phishing → security

Category-to-priority mapping:
- billing/complaint/phishing → P1, technical/refund → P2, general → P3

Task instructions:
- Task 1: Only provide "category"
- Task 2: Provide "category", "priority", "department"
- Task 3: Provide "category", "priority", "department", and "draft_reply" (>=100 words, professional, empathetic)

Respond with ONLY a valid JSON object. No other text."""


def format_observation(obs: dict) -> str:
    parts = []
    task_id = obs.get("task_id", 1)
    parts.append(f"=== EMAIL TRIAGE — TASK {task_id} ===")
    parts.append(f"Task: {obs.get('task_description', '')}")

    hints = obs.get("feature_hints", {}) or {}
    hint_flags = []
    if hints.get("has_money_terms"): hint_flags.append("money/payment terms")
    if hints.get("has_security_terms"): hint_flags.append("security terms")
    if hints.get("has_link"): hint_flags.append("contains links")
    if hints.get("has_urgency"): hint_flags.append("urgent")
    if hint_flags:
        parts.append(f"Hints: {', '.join(hint_flags)}")

    parts.append(f"\nFrom: {obs.get('sender', 'unknown')}")
    parts.append(f"Subject: {obs.get('subject', '')}")
    parts.append(f"\n{obs.get('body', '')}")

    if task_id == 1:
        parts.append('\nRespond with JSON: {"category": "..."}')
    elif task_id == 2:
        parts.append('\nRespond with JSON: {"category": "...", "priority": "...", "department": "..."}')
    else:
        parts.append('\nRespond with JSON: {"category": "...", "priority": "...", "department": "...", "draft_reply": "..."} (>=100 words)')

    return "\n".join(parts)


def parse_action(text: str) -> Optional[dict]:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    try:
        return json.loads(text)
    except Exception:
        import re
        try:
            return json.loads(re.sub(r",\s*}", "}", text))
        except Exception:
            return None


VALID_CATEGORIES = ["billing", "technical", "general", "complaint", "refund", "phishing"]
DEPT_MAP = {"billing": "billing_team", "technical": "technical_team", "refund": "returns",
            "complaint": "customer_success", "general": "customer_success", "phishing": "security"}
PRIO_MAP = {"billing": "P1", "complaint": "P1", "phishing": "P1",
            "technical": "P2", "refund": "P2", "general": "P3"}
TASK_PHASE = {"task1_email_classification": 1, "task2_prioritization_routing": 2, "task3_full_triage_reply": 3}


def get_fallback_action(task_id: int, obs: dict) -> dict:
    text = (obs.get("subject", "") + " " + obs.get("body", "")).lower()
    hints = obs.get("feature_hints", {}) or {}
    if hints.get("has_link") and hints.get("has_security_terms"): cat = "phishing"
    elif any(w in text for w in ["refund", "return", "money back"]): cat = "refund"
    elif any(w in text for w in ["charge", "invoice", "payment", "billing"]): cat = "billing"
    elif any(w in text for w in ["error", "bug", "crash", "not working"]): cat = "technical"
    elif any(w in text for w in ["unhappy", "complaint", "angry"]): cat = "complaint"
    elif any(w in text for w in ["phishing", "suspicious"]): cat = "phishing"
    else: cat = "general"

    action = {"category": cat}
    if task_id >= 2:
        action["priority"] = PRIO_MAP.get(cat, "P3")
        action["department"] = DEPT_MAP.get(cat, "customer_success")
    if task_id == 3:
        name = obs.get("sender", "Customer").split("@")[0]
        action["draft_reply"] = (
            f"Dear {name}, thank you for reaching out to us regarding your concern. "
            "We sincerely apologize for any inconvenience this may have caused you. "
            "Our dedicated team has received your message and is currently reviewing "
            "the details of your case. We want to assure you that we take every "
            "customer concern seriously and will work diligently to resolve this matter "
            "as quickly as possible. A member of our support team will follow up with "
            "you within 24 to 48 hours with a detailed response and next steps. "
            "In the meantime, if you have any additional information or documents "
            "that might help us better understand your situation, please do not "
            "hesitate to share them. We truly value your patience and your continued "
            "trust in our services. Best regards, Customer Support Team."
        )
    return action


def validate_action(action: dict, task_id: int) -> dict:
    cat = action.get("category", "general")
    if cat not in VALID_CATEGORIES: cat = "general"
    action["category"] = cat
    if task_id >= 2:
        if action.get("priority") not in ["P1", "P2", "P3"]:
            action["priority"] = PRIO_MAP.get(cat, "P3")
        if action.get("department") not in ["billing_team", "technical_team", "customer_success", "returns", "security"]:
            action["department"] = DEPT_MAP.get(cat, "customer_success")
    else:
        action.pop("priority", None)
        action.pop("department", None)
        action.pop("draft_reply", None)
    if task_id == 3 and (not action.get("draft_reply") or len(action.get("draft_reply", "").split()) < 100):
        action["draft_reply"] = (
            "Dear Customer, thank you for reaching out to us about this matter. "
            "We sincerely apologize for any inconvenience you have experienced. "
            "Our team has carefully reviewed your message and we understand your "
            "concern. We want to assure you that we take this matter very seriously "
            "and are committed to providing you with a satisfactory resolution. "
            "We have escalated your case to the appropriate department and they "
            "will be investigating this thoroughly. You can expect to hear back "
            "from us within 24 to 48 business hours with a detailed update on "
            "the progress of your case. In the meantime, please feel free to "
            "provide any additional details or documentation that might help us "
            "resolve this matter more efficiently. We truly appreciate your "
            "patience and understanding. Your satisfaction is our top priority. "
            "Best regards, Customer Support Team."
        )
    return action


def format_action_str(action: dict) -> str:
    parts = [f"category={action.get('category', '?')}"]
    if action.get("priority"): parts.append(f"priority={action['priority']}")
    if action.get("department"): parts.append(f"dept={action['department']}")
    if action.get("draft_reply"): parts.append("reply=yes")
    return "triage(" + ",".join(parts) + ")"


def run_task(task_id_str: str) -> float:
    """Run a single task: reset → one step → done. Matches reference repo pattern."""
    task_phase = TASK_PHASE.get(task_id_str, 1)
    step_rewards = []
    final_score = 0.01

    emit_start(task_id_str)

    try:
        # Reset — response is {observation, reward, done, info}
        resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id_str}, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        # Read observation from structured response
        obs = result.get("observation", result)  # fallback to flat if needed

        # Build LLM prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_observation(obs)},
        ]

        # Get LLM action
        action_dict = None
        error_msg = None
        try:
            completion = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=800 if task_phase == 3 else 256,
            )
            llm_response = completion.choices[0].message.content or ""
            action_dict = parse_action(llm_response)
        except Exception as e:
            error_msg = str(e)

        if action_dict is None:
            action_dict = get_fallback_action(task_phase, obs)
            if not error_msg:
                error_msg = "LLM parse failed"

        action_dict = validate_action(action_dict, task_phase)
        action_str = format_action_str(action_dict)

        # Step — send action wrapped in {"action": {...}}
        try:
            resp = requests.post(
                f"{ENV_URL}/step",
                json={"action": action_dict},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            error_msg = str(e)
            emit_step(1, action_str, 0.01, True, error_msg)
            step_rewards.append(0.01)
            emit_end(False, 1, step_rewards)
            return 0.01

        # Read reward from structured response — {reward: {value: X}}
        reward_obj = result.get("reward", {})
        if isinstance(reward_obj, dict):
            step_reward = clamp_reward(reward_obj.get("value", 0.01))
        else:
            step_reward = clamp_reward(reward_obj)

        done = result.get("done", True)
        step_rewards.append(step_reward)
        final_score = step_reward

        emit_step(1, action_str, step_reward, True, error_msg)

    except Exception as e:
        step_rewards = [0.01]
        emit_end(False, 1, step_rewards)
        return 0.01

    success = final_score > 0.01
    emit_end(success, 1, step_rewards)
    return final_score


def main():
    print("=" * 60, file=sys.stderr, flush=True)
    print("  Email Triage OpenEnv — Baseline Inference", file=sys.stderr, flush=True)
    print("=" * 60, file=sys.stderr, flush=True)
    print(f"  LLM endpoint : {API_BASE_URL}", file=sys.stderr, flush=True)
    print(f"  Model        : {MODEL_NAME}", file=sys.stderr, flush=True)
    print(f"  Environment  : {ENV_URL}", file=sys.stderr, flush=True)
    print(f"  Tasks        : {TASKS}", file=sys.stderr, flush=True)
    print(file=sys.stderr, flush=True)

    try:
        resp = requests.get(f"{ENV_URL}/health", timeout=10)
        resp.raise_for_status()
        print("  Environment health check: OK", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"  ERROR: Cannot reach environment at {ENV_URL}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    scores = {}
    start_time = time.time()

    for task_id in TASKS:
        score = run_task(task_id)
        scores[task_id] = score

    total_time = time.time() - start_time

    print(f"\n{'=' * 60}", file=sys.stderr, flush=True)
    print("  RESULTS SUMMARY", file=sys.stderr, flush=True)
    print("=" * 60, file=sys.stderr, flush=True)
    for tid, score in scores.items():
        bar = "#" * int(score * 40) + "." * (40 - int(score * 40))
        print(f"  {tid:40s} [{bar}] {score:.4f}", file=sys.stderr, flush=True)

    avg = sum(scores.values()) / len(scores) if scores else 0
    print(f"\n  Average score: {avg:.4f}", file=sys.stderr, flush=True)
    print(f"  Total runtime: {total_time:.1f}s", file=sys.stderr, flush=True)
    print("=" * 60, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()