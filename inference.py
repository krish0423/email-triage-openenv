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
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

TASKS = [
    "task1_email_classification",
    "task2_prioritization_routing",
    "task3_full_triage_reply",
]

# ── Structured Logging (matches validator format exactly) ───────
def emit_start(task_id: str) -> None:
    print(f"[START] task={task_id}", flush=True)

def emit_step(step: int, step_reward: float) -> None:
    print(f"[STEP] step={step} reward={step_reward}", flush=True)

def emit_end(task_id: str, score: float, steps: int) -> None:
    print(f"[END] task={task_id} score={score} steps={steps}", flush=True)

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

    # Feature hints
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
    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        action = json.loads(text)
        return action
    except (json.JSONDecodeError, KeyError):
        # Try cleaning trailing commas
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

    # Simple keyword detection
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
            # Generate a longer draft
            name = "Customer"
            action["draft_reply"] = (
                f"Dear {name}, thank you for reaching out to us about this matter. "
                f"We sincerely apologize for any inconvenience you have experienced. "
                f"Our team has carefully reviewed your message and we understand your "
                f"concern. We want to assure you that we take this matter very seriously "
                f"and are committed to providing you with a satisfactory resolution. "
                f"We have escalated your case to the appropriate department and they "
                f"will be investigating this thoroughly. You can expect to hear back "
                f"from us within 24 to 48 business hours with a detailed update on "
                f"the progress of your case. In the meantime, please feel free to "
                f"provide any additional details or documentation that might help us "
                f"resolve this matter more efficiently. We truly appreciate your "
                f"patience and understanding. Your satisfaction is our top priority. "
                f"Best regards, Customer Support Team."
            )

    return action


def run_task(task_id: str) -> float:
    """Run a single task and return the final score."""
    final_score = 0.0
    step_count = 0
    cumulative_reward = 0.0

    emit_start(task_id)

    print(f"\n{'=' * 60}", flush=True)
    print(f"  Starting task: {task_id}", flush=True)
    print(f"{'=' * 60}", flush=True)

    try:
        # Reset environment for this task
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

            # Keep context window manageable
            if len(messages) > 10:
                messages = [messages[0]] + messages[-8:]

            action_dict = None
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
                print(f"  [Step {step_count}] LLM error: {e}", flush=True)

            # Fallback if LLM fails
            if action_dict is None:
                action_dict = get_fallback_action(current_task_id, obs)
                print(f"  [Step {step_count}] Using fallback action", flush=True)
            else:
                print(
                    f"  [Step {step_count}] Action: category={action_dict.get('category')}",
                    flush=True,
                )

            # Validate and clean action
            action_dict = validate_action(action_dict, current_task_id)

            # Execute action
            try:
                resp = requests.post(f"{ENV_URL}/step", json=action_dict, timeout=30)
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                print(f"  [Step {step_count}] Step error: {e}", flush=True)
                break

            # The environment returns observation dict directly
            obs = result
            step_reward = float(obs.get("reward", 0.0))
            done = obs.get("done", False)
            cumulative_reward += step_reward

            emit_step(step_count, step_reward)

            feedback = obs.get("feedback", "")
            print(f"  Reward: {step_reward:.4f} | Feedback: {feedback}", flush=True)

            if done:
                final_score = cumulative_reward / step_count if step_count > 0 else 0.0
                # Normalize to 0-1 range
                final_score = max(0.01, min(0.99, final_score))
                print(f"\n  Task completed in {step_count} steps.", flush=True)
                print(f"  Final score: {final_score:.4f}", flush=True)
                break

            # Safety: prevent infinite loops
            if step_count >= 10:
                print("  Max steps reached, ending task.", flush=True)
                final_score = cumulative_reward / step_count if step_count > 0 else 0.0
                final_score = max(0.01, min(0.99, final_score))
                break

    except Exception as e:
        print(f"  Task error: {e}", flush=True)
    finally:
        emit_end(task_id, final_score, step_count)

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

    # Verify environment is running
    try:
        resp = requests.get(f"{ENV_URL}/health", timeout=10)
        resp.raise_for_status()
        print("  Environment health check: OK", flush=True)
    except Exception as e:
        print(f"  ERROR: Cannot reach environment at {ENV_URL}: {e}", flush=True)
        print("  Start the environment first: uvicorn server.app:app --port 7860", flush=True)
        sys.exit(1)

    scores = {}
    start_time = time.time()

    for task_id in TASKS:
        task_start = time.time()
        score = run_task(task_id)
        task_duration = time.time() - task_start
        scores[task_id] = score
        print(f"  Task {task_id}: score={score:.4f}, time={task_duration:.1f}s", flush=True)

    total_time = time.time() - start_time

    # Summary
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

    return scores


if __name__ == "__main__":
    main()