"""
Baseline inference script for the Email Triage OpenEnv environment.

Uses OpenAI-compatible API client (e.g. Groq) to power an LLM agent
that triages customer support emails through the OpenEnv API.

Required environment variables:
  API_BASE_URL — The API endpoint for the LLM (e.g. https://api.groq.com/openai/v1)
  MODEL_NAME   — The model identifier (e.g. llama-3.1-8b-instant)
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

# ── Structured Logging ──────────────────────────
# Official format:
#   [START] task=<task_name> env=<benchmark> model=<model_name>
#   [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
#   [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

def emit_start(task_id: str) -> None:
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

def emit_step(step: int, action_str: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    done_str = "true" if done else "false"
    error_str = error if error else "null"
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

def emit_end(success: bool, steps: int, rewards: list) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)

# ── LLM Client ──────────────────────────────────
llm_client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

SYSTEM_PROMPT = """You are an expert customer support email triage agent.

You will be shown a customer email and must triage it by providing a structured JSON action.

Available categories: billing, technical, general, complaint, refund, phishing
Available priorities: P1 (urgent), P2 (medium), P3 (low)
Available departments: billing_team, technical_team, customer_success, returns, security

Category-to-department mapping guide:
- billing → billing_team
- technical → technical_team
- complaint → customer_success
- general → customer_success
- refund → returns
- phishing → security

Category-to-priority mapping guide:
- billing → P1
- complaint → P1
- phishing → P1
- technical → P2
- refund → P2
- general → P3

You will receive task instructions indicating what fields to fill:
- Task 1: Only provide "category"
- Task 2: Provide "category", "priority", "department"
- Task 3: Provide "category", "priority", "department", and "draft_reply" (a professional, empathetic reply of at least 100 words)

IMPORTANT for draft_reply (Task 3):
- Write a professional, empathetic customer support reply
- Must be at least 100 words
- Address the customer's specific concern
- Do NOT repeat sentences
- If you detect phishing, warn the customer about security risks

Respond with ONLY a valid JSON object. No other text.

Example for Task 1:
{"category": "billing"}

Example for Task 2:
{"category": "billing", "priority": "P1", "department": "billing_team"}

Example for Task 3:
{"category": "billing", "priority": "P1", "department": "billing_team", "draft_reply": "Dear customer, ..."}"""

def format_observation(obs: dict) -> str:
    """Format an observation into a readable string for the LLM."""
    parts = []
    task_id = obs.get("task_id", 1)
    parts.append(f"=== EMAIL TRIAGE — TASK {task_id} ===")
    parts.append(f"Task: {obs.get('task_description', '')}")

    hints = obs.get("feature_hints", {}) or {}
    hint_flags = []
    if hints.get("has_money_terms"):
        hint_flags.append("contains money/payment terms")
    if hints.get("has_security_terms"):
        hint_flags.append("contains security-related terms")
    if hints.get("has_link"):
        hint_flags.append("contains links/URLs")
    if hints.get("has_urgency"):
        hint_flags.append("marked as urgent")
    if hint_flags:
        parts.append(f"Hints: {', '.join(hint_flags)}")

    parts.append(f"\nFrom: {obs.get('sender', 'unknown')}")
    parts.append(f"Subject: {obs.get('subject', '')}")
    parts.append(f"\n{obs.get('body', '')}")

    if task_id == 1:
        parts.append("\nRespond with JSON: {\"category\": \"...\"}")
    elif task_id == 2:
        parts.append("\nRespond with JSON: {\"category\": \"...\", \"priority\": \"...\", \"department\": \"...\"}")
    else:
        parts.append("\nRespond with JSON: {\"category\": \"...\", \"priority\": \"...\", \"department\": \"...\", \"draft_reply\": \"...\"}")

    return "\n".join(parts)


def parse_action(text: str) -> Optional[dict]:
    """Parse an LLM response into an action dict."""
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
    except (json.JSONDecodeError, KeyError):
        import re
        cleaned = re.sub(r",\s*}", "}", text)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            return None


VALID_CATEGORIES = ["billing", "technical", "general", "complaint", "refund", "phishing"]
DEPT_MAP = {
    "billing": "billing_team", "technical": "technical_team",
    "refund": "returns", "complaint": "customer_success",
    "general": "customer_success", "phishing": "security",
}
PRIO_MAP = {
    "billing": "P1", "complaint": "P1", "phishing": "P1",
    "technical": "P2", "refund": "P2", "general": "P3",
}


def get_fallback_action(task_id: int, obs: dict) -> dict:
    """Heuristic fallback if LLM parsing fails."""
    text = (obs.get("subject", "") + " " + obs.get("body", "")).lower()
    hints = obs.get("feature_hints", {}) or {}

    if hints.get("has_link") and hints.get("has_security_terms"):
        cat = "phishing"
    elif any(w in text for w in ["refund", "return", "money back"]):
        cat = "refund"
    elif any(w in text for w in ["charge", "invoice", "payment", "billing"]):
        cat = "billing"
    elif any(w in text for w in ["error", "bug", "crash", "not working"]):
        cat = "technical"
    elif any(w in text for w in ["unhappy", "complaint", "angry", "disappointed"]):
        cat = "complaint"
    elif any(w in text for w in ["phishing", "suspicious", "click the link"]):
        cat = "phishing"
    else:
        cat = "general"

    action = {"category": cat}
    if task_id >= 2:
        action["priority"] = PRIO_MAP.get(cat, "P3")
        action["department"] = DEPT_MAP.get(cat, "customer_success")
    if task_id == 3:
        name = obs.get("sender", "Customer").split("@")[0]
        subject = obs.get("subject", "your request")
        action["draft_reply"] = (
            f"Dear {name}, thank you for reaching out to us regarding {subject}. "
            f"We sincerely apologize for any inconvenience this may have caused you. "
            f"Our dedicated team has received your message and is currently reviewing "
            f"the details of your case. We want to assure you that we take every "
            f"customer concern seriously and will work diligently to resolve this matter "
            f"as quickly as possible. A member of our support team will follow up with "
            f"you within 24 to 48 hours with a detailed response and next steps. "
            f"In the meantime, if you have any additional information or documents "
            f"that might help us better understand your situation, please do not "
            f"hesitate to share them. We truly value your patience and your continued "
            f"trust in our services. Best regards, Customer Support Team."
        )
    return action


def validate_action(action: dict, task_id: int) -> dict:
    """Ensure action has valid fields for the task."""
    cat = action.get("category", "general")
    if cat not in VALID_CATEGORIES:
        cat = "general"
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

    if task_id == 3:
        draft = action.get("draft_reply", "")
        if not draft or len(draft.split()) < 100:
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


def parse_and_validate_action(llm_text: str, task_id: int, obs: dict) -> dict:
    """
    Robust wrapper: parse LLM text, sanitize fields, and guarantee a valid action dict.
    Uses parse_action(), get_fallback_action(), and validate_action() already defined.
    """
    # 1) Try to parse raw LLM output
    action = parse_action(llm_text) if llm_text else None

    # 2) If parse failed, use fallback heuristic
    if not action:
        action = get_fallback_action(task_id, obs)

    # 3) Normalize keys and values
    if "category" in action and isinstance(action["category"], str):
        action["category"] = action["category"].strip().lower()
    if "priority" in action and isinstance(action["priority"], str):
        action["priority"] = action["priority"].strip().upper()
    if "department" in action and isinstance(action["department"], str):
        action["department"] = action["department"].strip().lower()

    # 4) Map common department synonyms to canonical names
    dept_synonyms = {
        "support": "customer_success", "support_team": "customer_success",
        "tech": "technical_team", "tech_team": "technical_team",
        "billing": "billing_team", "accounts": "billing_team",
        "security": "security", "fraud": "security", "returns": "returns"
    }
    if "department" in action and action["department"] in dept_synonyms:
        action["department"] = dept_synonyms[action["department"]]

    # 5) If category missing or invalid, infer from text
    if not action.get("category") or action["category"] not in VALID_CATEGORIES:
        inferred = get_fallback_action(task_id, obs).get("category")
        action["category"] = inferred

    # 6) Ensure priority/department/draft_reply fields are present/valid for the task
    action = validate_action(action, task_id)

    # 7) Final safety: ensure no empty strings for optional fields
    for k in ["priority", "department"]:
        if action.get(k) == "" or action.get(k) is None:
            if task_id >= 2:
                action[k] = PRIO_MAP.get(action["category"], "P3") if k == "priority" else DEPT_MAP.get(action["category"], "customer_success")
            else:
                action.pop(k, None)

    return action


def format_action_str(action: dict) -> str:
    """Format action dict into a compact string for the [STEP] log line."""
    cat = action.get("category", "unknown")
    prio = action.get("priority")
    dept = action.get("department")
    has_reply = bool(action.get("draft_reply"))

    parts = [f"category={cat}"]
    if prio:
        parts.append(f"priority={prio}")
    if dept:
        parts.append(f"dept={dept}")
    if has_reply:
        parts.append("reply=yes")
    return "triage(" + ",".join(parts) + ")"


def run_task(task_id: str) -> float:
    """Run a single task and return the final score."""
    final_score = 0.0
    step_count = 0
    cumulative_reward = 0.0
    step_rewards = []

    emit_start(task_id)

    try:
        resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        done = obs.get("done", False)

        while not done:
            step_count += 1
            current_task_id = obs.get("task_id", 1)

            obs_text = format_observation(obs)
            messages.append({"role": "user", "content": obs_text})

            if len(messages) > 10:
                messages = [messages[0]] + messages[-8:]

            action_dict = None
            error_msg = None
            llm_response = ""
            try:
                completion = llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=800 if current_task_id == 3 else 256,
                )
                llm_response = completion.choices[0].message.content or ""
                messages.append({"role": "assistant", "content": llm_response})

                # Use the robust parser + validator
                action_dict = parse_and_validate_action(llm_response, current_task_id, obs)

            except Exception as e:
                error_msg = str(e)

            if action_dict is None:
                # As a final fallback (shouldn't happen because parse_and_validate_action guarantees a dict)
                action_dict = get_fallback_action(current_task_id, obs)
                if error_msg is None:
                    error_msg = "LLM parse failed, using fallback"

            action_str = format_action_str(action_dict)

            try:
                resp = requests.post(f"{ENV_URL}/step", json=action_dict, timeout=30)
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                error_msg = str(e)
                emit_step(step_count, action_str, 0.0, True, error_msg)
                step_rewards.append(0.0)
                break

            obs = result
            step_reward = float(obs.get("reward", 0.0))
            done = obs.get("done", False)
            cumulative_reward += step_reward
            step_rewards.append(step_reward)

            emit_step(step_count, action_str, step_reward, done, error_msg)

            if done:
                final_score = cumulative_reward / step_count if step_count > 0 else 0.0
                final_score = max(0.01, min(0.99, final_score))
                break

            if step_count >= 10:
                final_score = cumulative_reward / step_count if step_count > 0 else 0.0
                final_score = max(0.01, min(0.99, final_score))
                break

    except Exception as e:
        if step_count == 0:
            step_count = 1
            step_rewards.append(0.0)

    finally:
        success = final_score > 0.0 and len(step_rewards) > 0
        emit_end(success, step_count, step_rewards)

    return final_score


def main():
    print("=" * 60, flush=True)
    print("  Email Triage OpenEnv — Baseline Inference", flush=True)
    print("=" * 60, flush=True)
    print(f"  LLM endpoint : {API_BASE_URL}", flush=True)
    print(f"  Model        : {MODEL_NAME}", flush=True)
    print(f"  Environment  : {ENV_URL}", flush=True)
    print(f"  Tasks        : {TASKS}", flush=True)
    print(flush=True)

    try:
        resp = requests.get(f"{ENV_URL}/health", timeout=10)
        resp.raise_for_status()
        print("  Environment health check: OK", flush=True)
    except Exception as e:
        print(f"  ERROR: Cannot reach environment at {ENV_URL}: {e}", flush=True)
        sys.exit(1)

    scores = {}
    start_time = time.time()

    for task_id in TASKS:
        task_start = time.time()
        score = run_task(task_id)
        task_duration = time.time() - task_start
        scores[task_id] = score

    total_time = time.time() - start_time

    print(f"\n{'=' * 60}", flush=True)
    print("  RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for tid, score in scores.items():
        bar = "#" * int(score * 40) + "." * (40 - int(score * 40))
        print(f"  {tid:40s} [{bar}] {score:.4f}", flush=True)

    avg_score = sum(scores.values()) / len(scores) if scores else 0
    print(f"\n  Average score: {avg_score:.4f}", flush=True)
    print(f"  Total runtime: {total_time:.1f}s", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
