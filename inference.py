"""
Baseline inference script for the Email Triage OpenEnv environment.

Uses OpenAI-compatible API client (e.g. Groq) to power an LLM agent
that triages customer support emails through the OpenEnv API.

Required environment variables:
  API_BASE_URL — The API endpoint for the LLM (default: Groq)
  MODEL_NAME   — The model identifier (default: llama-3.1-8b-instant)
  HF_TOKEN     — API key for the LLM provider (Groq: gsk_...)
  ENV_URL      — Environment server URL (default: http://localhost:7860)

Usage:
  cp .env.example .env          # fill in your Groq key
  python inference.py
"""

import os
import sys
import json
import time
import requests
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file if present
load_dotenv()

# ── Configuration ───────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.1-8b-instant")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable is required", flush=True)
    sys.exit(1)

TASKS = [
    "task1_email_classification",
    "task2_prioritization_routing",
    "task3_full_triage_reply",
]

TASK_PHASE = {
    "task1_email_classification":  1,
    "task2_prioritization_routing": 2,
    "task3_full_triage_reply":      3,
}

ENV_NAME = "email-triage-env"


# ── Reward helpers ───────────────────────────────────────────────────────────
def clamp_reward(r: float) -> float:
    """Keep score strictly inside (0, 1) — 0.0 and 1.0 are rejected by evaluator."""
    return max(0.01, min(0.99, float(r)))


# ── Structured logging (stdout — evaluator reads stdout) ─────────────────────
# ── Structured logging (stdout — evaluator reads stdout) ─────────────────────
def emit_start(task_id: str) -> None:
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)


def emit_step(step: int, action_dict: dict, reward: float, done: bool, error: Optional[str] = None) -> None:
    action_str = json.dumps(action_dict, separators=(",", ":"))
    done_str   = "true" if done else "false"
    error_str  = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={clamp_reward(reward):.2f} done={done_str} error={error_str}",
        flush=True,
    )


def emit_end(success: bool, steps: int, score: float, rewards: list) -> None:
    success_str  = "true" if success else "false"
    score_str    = f"{float(score):.2f}"
    rewards_str  = ",".join(f"{clamp_reward(r):.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score_str} rewards={rewards_str}", flush=True)


# ── LLM client ───────────────────────────────────────────────────────────────
llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are an expert customer support email triage agent.

You will be shown a customer support email and must triage it with a structured JSON action.

Available categories : billing, technical, general, complaint, refund, phishing
Available priorities : P1 (urgent), P2 (medium), P3 (low)
Available departments: billing_team, technical_team, customer_success, returns, security

Category → department:
  billing   → billing_team
  technical → technical_team
  complaint → customer_success
  general   → customer_success
  refund    → returns
  phishing  → security

Category → priority:
  billing / complaint / phishing → P1
  technical / refund             → P2
  general                        → P3

Phishing signals: urgent account alerts, suspicious links, requests for credentials or payment outside normal flow.

Task instructions
  Task 1 — provide ONLY "category"
  Task 2 — provide "category", "priority", "department"
  Task 3 — provide "category", "priority", "department", and "draft_reply"
            (draft_reply must be ≥100 words, professional, empathetic, and relevant to the email)

Respond with ONLY a valid JSON object. No explanation, no markdown, no extra text."""


# ── Observation formatter ─────────────────────────────────────────────────────
def format_observation(obs: dict, prev_feedback: str = "") -> str:
    parts = []
    task_id = obs.get("task_id", 1)
    parts.append(f"=== EMAIL TRIAGE — TASK {task_id} ===")
    parts.append(f"Task: {obs.get('task_description', '')}")

    if prev_feedback:
        parts.append(f"Feedback from last action: {prev_feedback}")

    hints = obs.get("feature_hints", {}) or {}
    hint_flags = []
    if hints.get("has_money_terms"):    hint_flags.append("money/payment terms detected")
    if hints.get("has_security_terms"): hint_flags.append("security terms detected")
    if hints.get("has_link"):           hint_flags.append("contains links")
    if hints.get("has_urgency"):        hint_flags.append("marked urgent")
    if hints.get("multi_intent"):       hint_flags.append("multiple intents")
    if hint_flags:
        parts.append(f"Hints: {', '.join(hint_flags)}")

    parts.append(f"\nFrom   : {obs.get('sender', 'unknown')}")
    parts.append(f"Subject: {obs.get('subject', '')}")
    parts.append(f"\n{obs.get('body', '')}")

    if task_id == 1:
        parts.append('\nRespond: {"category": "..."}')
    elif task_id == 2:
        parts.append('\nRespond: {"category": "...", "priority": "...", "department": "..."}')
    else:
        parts.append('\nRespond: {"category": "...", "priority": "...", "department": "...", "draft_reply": "..."} (>=100 words)')

    return "\n".join(parts)


# ── Action parsing ────────────────────────────────────────────────────────────
def parse_action(text: str) -> Optional[dict]:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
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


# ── Validation constants ──────────────────────────────────────────────────────
VALID_CATEGORIES = ["billing", "technical", "general", "complaint", "refund", "phishing"]
VALID_PRIORITIES = ["P1", "P2", "P3"]
VALID_DEPARTMENTS = ["billing_team", "technical_team", "customer_success", "returns", "security"]

DEPT_MAP = {
    "billing":   "billing_team",
    "technical": "technical_team",
    "refund":    "returns",
    "complaint": "customer_success",
    "general":   "customer_success",
    "phishing":  "security",
}
PRIO_MAP = {
    "billing":   "P1", "complaint": "P1", "phishing": "P1",
    "technical": "P2", "refund":    "P2",
    "general":   "P3",
}


# ── Fallback action (rule-based, used when LLM parse fails) ──────────────────
def get_fallback_action(task_id: int, obs: dict) -> dict:
    text  = (obs.get("subject", "") + " " + obs.get("body", "")).lower()
    hints = obs.get("feature_hints", {}) or {}

    if hints.get("has_link") and hints.get("has_security_terms"):
        cat = "phishing"
    elif any(w in text for w in ["refund", "return", "money back"]):
        cat = "refund"
    elif any(w in text for w in ["charge", "invoice", "payment", "billing"]):
        cat = "billing"
    elif any(w in text for w in ["error", "bug", "crash", "not working", "broken"]):
        cat = "technical"
    elif any(w in text for w in ["unhappy", "disappointed", "complaint", "angry", "frustrated"]):
        cat = "complaint"
    elif any(w in text for w in ["phishing", "suspicious", "verify your account", "click here"]):
        cat = "phishing"
    else:
        cat = "general"

    action: dict = {"category": cat}

    if task_id >= 2:
        action["priority"]   = PRIO_MAP.get(cat, "P3")
        action["department"] = DEPT_MAP.get(cat, "customer_success")

    if task_id == 3:
        name = obs.get("sender", "Customer").split("@")[0].replace(".", " ").title()
        action["draft_reply"] = (
            f"Dear {name}, thank you for reaching out to our support team. "
            "We have received your message and want to assure you that it has been "
            "reviewed carefully by our team. We sincerely apologize for any "
            "inconvenience or frustration this situation may have caused you. "
            "Your concern is extremely important to us and we are committed to "
            "providing you with a swift and satisfactory resolution. Our dedicated "
            "support team has escalated your case to the appropriate department, "
            "who will be investigating this matter thoroughly on your behalf. "
            "You can expect to hear back from us within 24 to 48 business hours "
            "with a detailed update and the next steps we will be taking. "
            "If you have any additional information, screenshots, or documents "
            "that may help us better understand or resolve your issue, please do "
            "not hesitate to reply to this message. We truly value your patience "
            "and your continued trust in our services. "
            "Best regards, Customer Support Team."
        )

    return action


# ── Action validation ─────────────────────────────────────────────────────────
def validate_action(action: dict, task_id: int) -> dict:
    cat = action.get("category", "general")
    if cat not in VALID_CATEGORIES:
        cat = "general"
    action["category"] = cat

    if task_id >= 2:
        if action.get("priority") not in VALID_PRIORITIES:
            action["priority"] = PRIO_MAP.get(cat, "P3")
        if action.get("department") not in VALID_DEPARTMENTS:
            action["department"] = DEPT_MAP.get(cat, "customer_success")
    else:
        action.pop("priority",    None)
        action.pop("department",  None)
        action.pop("draft_reply", None)

    if task_id == 3:
        reply = action.get("draft_reply", "")
        if not reply or len(reply.split()) < 100:
            action["draft_reply"] = (
                "Dear Customer, thank you for reaching out to us about this matter. "
                "We sincerely apologize for any inconvenience you have experienced. "
                "Our team has carefully reviewed your message and we understand your "
                "concern. We want to assure you that we take this matter very seriously "
                "and are fully committed to providing you with a satisfactory resolution. "
                "We have escalated your case to the appropriate department and they "
                "will be investigating this thoroughly on your behalf. You can expect "
                "to hear back from us within 24 to 48 business hours with a detailed "
                "update on the progress of your case and the next steps we will take. "
                "In the meantime, please feel free to provide any additional details "
                "or documentation that might help us resolve this matter more efficiently. "
                "We truly appreciate your patience and understanding throughout this process. "
                "Your satisfaction is our top priority and we are here to help. "
                "Best regards, Customer Support Team."
            )

    return action


# ── Task runner ───────────────────────────────────────────────────────────────
def run_task(task_id_str: str) -> float:
    """
    Reset the environment for a specific task, then loop until done=True,
    maintaining multi-turn conversation history across steps.
    Returns the clamped final score.
    """
    task_phase   = TASK_PHASE.get(task_id_str, 1)
    step_count   = 0
    final_score  = 0.01
    step_rewards = []
    prev_feedback = ""

    emit_start(task_id_str)

    # ── Reset ────────────────────────────────────────────────────────────────
    try:
        resp = requests.post(
            f"{ENV_URL}/reset",
            json={"task_id": task_id_str},
            timeout=30,
        )
        resp.raise_for_status()
        reset_result = resp.json()
    except Exception as e:
        print(f"  [reset error] {e}", file=sys.stderr, flush=True)
        emit_end(success=False, steps=0, score=0.01, rewards=[0.01])
        return 0.01

    # Support both flat and nested observation formats
    obs  = reset_result.get("observation", reset_result)
    done = reset_result.get("done", False)

    # Multi-turn conversation history (system + rolling window)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ── Step loop ────────────────────────────────────────────────────────────
    while not done:
        step_count += 1

        # Use task_id from observation if available (handles progressive tasks)
        current_task_id = obs.get("task_id", task_phase)
        if isinstance(current_task_id, str):
            current_task_id = TASK_PHASE.get(current_task_id, task_phase)

        # Build user message and append to history
        user_msg = format_observation(obs, prev_feedback=prev_feedback)
        messages.append({"role": "user", "content": user_msg})

        # Cap context window: keep system prompt + last 12 messages
        if len(messages) > 14:
            messages = [messages[0]] + messages[-12:]

        # ── LLM call ─────────────────────────────────────────────────────────
        action_dict = None
        error_msg   = None
        try:
            completion = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=800 if current_task_id == 3 else 256,
            )
            llm_response = completion.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": llm_response})
            action_dict = parse_action(llm_response)
        except Exception as e:
            error_msg = str(e)
            print(f"  [Step {step_count}] LLM error: {error_msg}", file=sys.stderr, flush=True)

        # Fall back to rule-based action if LLM fails or returns unparseable output
        if action_dict is None:
            action_dict = get_fallback_action(current_task_id, obs)
            if not error_msg:
                error_msg = "LLM parse failed — using fallback"
            print(f"  [Step {step_count}] Fallback: {action_dict.get('category')}", file=sys.stderr, flush=True)

        action_dict = validate_action(action_dict, current_task_id)

        # ── Execute step ──────────────────────────────────────────────────────
        try:
            resp = requests.post(
                f"{ENV_URL}/step",
                json={"action": action_dict},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            print(f"  [Step {step_count}] Step error: {e}", file=sys.stderr, flush=True)
            emit_step(step_count, action_dict or {}, 0.01, done=True, error=str(e))
            step_rewards.append(0.01)
            break

        # ── Parse result ──────────────────────────────────────────────────────
        reward_obj  = result.get("reward", {})
        if isinstance(reward_obj, dict):
            raw_reward = float(reward_obj.get("value", 0.01))
        else:
            raw_reward = float(reward_obj or 0.01)

        obs           = result.get("observation", obs)
        done          = result.get("done", True)
        prev_feedback = obs.get("feedback", "") if isinstance(obs, dict) else ""

        step_reward = clamp_reward(raw_reward)
        step_rewards.append(step_reward)
        final_score = step_reward  # last step reward = episode score

        emit_step(step_count, action_dict, step_reward, done, error_msg)

        print(
            f"  [Step {step_count}] category={action_dict.get('category')} "
            f"reward={step_reward:.4f} done={done}",
            file=sys.stderr, flush=True,
        )

    emit_end(success=len(step_rewards) > 0, steps=step_count, score=final_score, rewards=step_rewards)
    return final_score


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    # Human-readable header → stderr (keeps stdout clean for evaluator)
    print("=" * 60,                                       file=sys.stderr, flush=True)
    print("  Email Triage OpenEnv — Baseline Inference",  file=sys.stderr, flush=True)
    print("=" * 60,                                       file=sys.stderr, flush=True)
    print(f"  LLM endpoint : {API_BASE_URL}",             file=sys.stderr, flush=True)
    print(f"  Model        : {MODEL_NAME}",               file=sys.stderr, flush=True)
    print(f"  Environment  : {ENV_URL}",                  file=sys.stderr, flush=True)
    print(f"  Tasks        : {TASKS}",                    file=sys.stderr, flush=True)
    print("",                                             file=sys.stderr, flush=True)

    # Health check
    try:
        resp = requests.get(f"{ENV_URL}/health", timeout=10)
        resp.raise_for_status()
        print("  Environment health check: OK", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"  ERROR: Cannot reach environment at {ENV_URL}: {e}", file=sys.stderr, flush=True)
        print("  Start the server first: uvicorn server.app:app --port 7860", file=sys.stderr, flush=True)
        sys.exit(1)

    scores: dict = {}
    start_time = time.time()

    for task_id in TASKS:
        print(f"\n{'─' * 60}", file=sys.stderr, flush=True)
        print(f"  Running: {task_id}", file=sys.stderr, flush=True)
        print(f"{'─' * 60}", file=sys.stderr, flush=True)

        task_start = time.time()
        score      = run_task(task_id)
        task_time  = time.time() - task_start

        scores[task_id] = score
        print(f"  Finished: score={score:.4f}  time={task_time:.1f}s", file=sys.stderr, flush=True)

    total_time = time.time() - start_time

    # Results summary → stderr
    print(f"\n{'=' * 60}",    file=sys.stderr, flush=True)
    print("  RESULTS SUMMARY", file=sys.stderr, flush=True)
    print("=" * 60,            file=sys.stderr, flush=True)
    for tid, score in scores.items():
        bar = "#" * int(score * 40) + "." * (40 - int(score * 40))
        print(f"  {tid:40s} [{bar}] {score:.4f}", file=sys.stderr, flush=True)

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n  Average score : {avg:.4f}",  file=sys.stderr, flush=True)
    print(f"  Total runtime : {total_time:.1f}s", file=sys.stderr, flush=True)
    print("=" * 60,                          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()