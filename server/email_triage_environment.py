# env.py — Email Triage Environment (final, pydantic-safe)
# - Preserves internal negative signal via raw_reward clipped to [-1.0, 1.0]
# - Returns external reward clamped to (0.01, 0.99) — strictly between 0 and 1
# - Final step returns the normalized cumulative score for the whole task
# - Instance-local RNG for reproducibility
# - Repetition penalty for draft replies
# - Logs both raw_reward and external_reward in JSONL
# - Hard-negative replay with limits
# - Basic action validation and defensive guards

import uuid
import random
import json
import time
import re
from typing import Tuple, Optional, Dict, Any

from models import TriageAction, TriageObservation, TriageState
from server.email_dataset import load_dataset, DatasetEnvHelper
from server.llm_judge import llm_judge_score

# -------------------------
# Configuration / Defaults
# -------------------------
REPLAY_RETRY_LIMIT = 3
PHISHING_BIAS_BASE = 0.1
PHISHING_BIAS_STEP = 0.05
PHISHING_BIAS_MAX = 0.7

EPSILON_SEED = 42
LOG_PATH = "env_step_logs.jsonl"

VALID_CATEGORIES = {"billing", "technical", "account", "phishing", "general", "refund", "complaint"}
VALID_PRIORITIES = {"P1", "P2", "P3", None}
VALID_DEPARTMENTS = {"billing_team", "technical_team", "customer_success", "security", "returns", None}

# -------------------------
# Load and normalize dataset
# -------------------------
DATASET = load_dataset()
ENV_HELPER = DatasetEnvHelper(DATASET)

def _normalize_ground_truth(dataset):
    for i, e in enumerate(dataset):
        gt = e.get("ground_truth", {}) or {}
        gt.setdefault("category", e.get("true_category"))
        gt.setdefault("priority", e.get("true_priority"))
        gt.setdefault("department", e.get("true_department"))
        gt.setdefault("disguised", bool(e.get("disguised", False) or e.get("is_adversarial", False)))
        e["ground_truth"] = gt
        if "email_id" not in e:
            e["email_id"] = e.get("id") or f"email_{i}"
        e.setdefault("sender", e.get("from", "unknown@domain.com"))
    return dataset

DATASET = _normalize_ground_truth(DATASET)
ENV_HELPER = DatasetEnvHelper(DATASET)

# -------------------------
# Utility helpers
# -------------------------
def _safe_find_email_by_id(email_id: str) -> Optional[Dict[str, Any]]:
    for e in DATASET:
        if e.get("email_id") == email_id:
            return e
    return None

def _normalize_llm_score(score: float, scale: float = 5.0) -> float:
    try:
        s = float(score)
    except Exception:
        return 0.0
    s = max(0.0, min(scale, s))
    return s / scale

def _log_step(entry: Dict[str, Any]):
    entry["timestamp"] = time.time()
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _validate_action_fields(action: TriageAction) -> Tuple[bool, str]:
    try:
        cat = getattr(action, "category", None)
        pr = getattr(action, "priority", None)
        dept = getattr(action, "department", None)
    except Exception:
        return False, "Action missing attributes"
    if cat is not None and cat not in VALID_CATEGORIES:
        return False, f"Invalid category '{cat}'"
    if pr not in VALID_PRIORITIES:
        return False, f"Invalid priority '{pr}'"
    if dept not in VALID_DEPARTMENTS:
        return False, f"Invalid department '{dept}'"
    return True, "ok"

def _sentence_split(text: str) -> list:
    return [s.strip() for s in re.split(r'[.!?]\s*', text) if s.strip()]

def _clamp(r: float) -> float:
    """Clamp reward to strictly open interval (0, 1). Safe for 2dp formatting."""
    return round(max(0.01, min(0.99, float(r))), 4)

# -------------------------
# Environment
# -------------------------
class EmailTriageEnvironment:
    def __init__(self, seed: Optional[int] = None):
        self._state: Optional[TriageState] = None
        self._current_email: Optional[Dict[str, Any]] = None
        self._index = 0
        self.history = []
        self._replay_counts = {}
        self._last_email_id: Optional[str] = None
        self._rng = random.Random(seed if seed is not None else EPSILON_SEED)
        # Accumulate raw scores per step within an episode for final normalization
        self._episode_raw_scores = []

    def reset(self, phase: int = 1, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self._rng = random.Random(seed)

        self._episode_raw_scores = []

        phishing_bias = min(PHISHING_BIAS_MAX, PHISHING_BIAS_BASE + self._index * PHISHING_BIAS_STEP)
        if phase >= 3:
            phishing_bias = min(PHISHING_BIAS_MAX, phishing_bias + 0.15)

        if self._rng.random() < phishing_bias:
            phishing_pool = [e for e in DATASET if e["ground_truth"].get("disguised")]
            if phishing_pool:
                self._current_email = self._rng.choice(phishing_pool)
            else:
                try:
                    self._current_email = ENV_HELPER.pop_next()
                except Exception:
                    self._current_email = self._rng.choice(DATASET)
        else:
            try:
                sample = ENV_HELPER.pop_next()
                self._current_email = sample
            except Exception:
                self._current_email = self._rng.choice(DATASET)

        self._index += 1

        if self.history:
            last = self.history[-1]
            if last.get("raw_reward", 0.0) < 0.3:
                failed = _safe_find_email_by_id(last.get("email_id"))
                if failed:
                    cnt = self._replay_counts.get(failed["email_id"], 0)
                    if cnt < REPLAY_RETRY_LIMIT:
                        if self._current_email is None or failed["email_id"] != self._current_email.get("email_id"):
                            self._current_email = failed
                            self._replay_counts[failed["email_id"]] = cnt + 1

        self._state = TriageState(
            episode_id=str(uuid.uuid4()),
            current_task_id=1,
            total_reward=0.0,
            steps=0,
            completed=False
        )

        _log_step({
            "event": "reset",
            "episode_id": self._state.episode_id,
            "email_id": self._current_email.get("email_id") if self._current_email else None,
            "phase": phase,
            "phishing_bias": phishing_bias
        })

        self._last_email_id = self._current_email.get("email_id") if self._current_email else None

        return self._make_observation(
            reward=0.01,
            raw_reward=0.0,
            done=False,
            feedback="New episode started. Use caution for phishing signals."
        )

    def step(self, action: TriageAction) -> Dict[str, Any]:
        if self._state is None or self._current_email is None:
            raise RuntimeError("Call reset() before step()")

        valid, msg = _validate_action_fields(action)
        if not valid:
            raw_reward = -0.5
            feedback = f"Invalid action: {msg}"
            self._state.steps += 1
            self._episode_raw_scores.append(raw_reward)

            task_id = self._state.current_task_id
            if task_id < 3:
                self._state.current_task_id += 1
                done = False
                external_reward = _clamp(raw_reward)
            else:
                done = True
                self._state.completed = True
                # On final step, compute normalized cumulative score
                avg = sum(self._episode_raw_scores) / len(self._episode_raw_scores)
                external_reward = _clamp(avg)

            self._state.total_reward += external_reward
            _log_step({
                "event": "step_invalid_action",
                "episode_id": self._state.episode_id,
                "email_id": self._current_email.get("email_id"),
                "step": self._state.steps,
                "task_id": self._state.current_task_id,
                "action": str(action),
                "raw_reward": round(raw_reward, 4),
                "reward": external_reward,
                "feedback": [feedback]
            })
            self.history.append({
                "timestamp": time.time(),
                "episode_id": self._state.episode_id,
                "email_id": self._current_email.get("email_id"),
                "action": action.dict() if hasattr(action, "dict") else getattr(action, "__dict__", str(action)),
                "raw_reward": round(raw_reward, 4),
                "reward": external_reward,
                "task_id": self._state.current_task_id,
                "reason_tags": ["invalid_action"]
            })
            return self._make_observation(reward=external_reward, raw_reward=raw_reward, done=done, feedback=feedback)

        task_id = self._state.current_task_id
        gt = self._current_email["ground_truth"]
        email = self._current_email

        reward = 0.0
        feedback_items = []
        reason_tags = []
        repetition_ratio = 0.0
        rep_penalty = 0.0

        if action.category == gt["category"]:
            reward += 0.4
            feedback_items.append("Correct category")
            reason_tags.append("category_match")
        else:
            reward -= 0.2
            feedback_items.append(f"Wrong category (expected {gt['category']})")
            reason_tags.append("category_mismatch")

        if gt.get("disguised"):
            if action.category == "phishing":
                reward += 0.5
                feedback_items.append("Correct phishing detection")
                reason_tags.append("phishing_detected")
            else:
                reward -= 0.5
                feedback_items.append("Missed phishing attack")
                reason_tags.append("phishing_missed")

        if gt["category"] == "phishing" and action.category != "phishing":
            text = (email.get("subject", "") + " " + email.get("body", "")).lower()
            signals = sum([
                "http" in text,
                "urgent" in text,
                any(k in text for k in ["password", "card", "bank", "verify"])
            ])
            if signals >= 2:
                reward += 0.3
                feedback_items.append("Partial phishing detection")
                reason_tags.append("partial_phishing")

        if task_id >= 2:
            if action.priority == gt.get("priority"):
                reward += 0.2
                feedback_items.append("Priority correct")
                reason_tags.append("priority_match")
            else:
                feedback_items.append("Priority incorrect")
                reason_tags.append("priority_mismatch")

            if action.department == gt.get("department"):
                reward += 0.2
                feedback_items.append("Department correct")
                reason_tags.append("department_match")
            else:
                feedback_items.append("Department incorrect")
                reason_tags.append("department_mismatch")

        if task_id == 3:
            draft = (action.draft_reply or "").strip()
            word_count = len(draft.split())
            if word_count < 100:
                feedback_items.append(f"Reply too short ({word_count} words); requires >=100 words")
                reward -= 0.1
                reason_tags.append("reply_too_short")
            else:
                sentences = _sentence_split(draft)
                if sentences:
                    unique = len(set(sentences))
                    total = len(sentences)
                    repetition_ratio = 1.0 - (unique / total)
                    rep_penalty = min(0.2, repetition_ratio * 0.5)
                    if rep_penalty > 0:
                        reward -= rep_penalty
                        feedback_items.append(f"Repetition penalty applied ({repetition_ratio:.2f})")
                        reason_tags.append("reply_repetition")
                if gt.get("disguised"):
                    keywords = ["phishing", "security", "do not", "report", "suspicious", "link"]
                    hits = sum(1 for k in keywords if k in draft.lower())
                    reward += min(0.2, hits * 0.05)
                    feedback_items.append(f"Reply phishing awareness keywords: {hits}")
                    reason_tags.append("reply_phishing_keywords")
                else:
                    try:
                        score, reason = llm_judge_score(email["subject"], email["body"], draft)
                        norm = _normalize_llm_score(score, scale=1.0)
                        reward += norm * 0.4
                        feedback_items.append(f"Reply quality (LLM): {norm:.3f} ({reason})")
                        reason_tags.append("llm_judge")
                    except Exception:
                        reward += 0.1
                        feedback_items.append("Reply judged by fallback")
                        reason_tags.append("llm_fallback")

        reward += self._rng.uniform(-0.02, 0.02)

        raw_reward = max(-1.0, min(1.0, reward))
        raw_reward = round(raw_reward, 4)

        # Accumulate raw score for this step
        self._episode_raw_scores.append(raw_reward)

        self._state.steps += 1

        if task_id < 3:
            self._state.current_task_id += 1
            done = False
            # For intermediate steps, clamp the individual step reward
            external_reward = _clamp(raw_reward)
        else:
            done = True
            self._state.completed = True
            # On final step, compute normalized cumulative score for the whole task
            # This ensures the task-level score is strictly in (0, 1)
            avg = sum(self._episode_raw_scores) / len(self._episode_raw_scores)
            external_reward = _clamp(avg)

        self._state.total_reward += external_reward

        self.history.append({
            "timestamp": time.time(),
            "episode_id": self._state.episode_id,
            "email_id": email.get("email_id"),
            "action": action.dict() if hasattr(action, "dict") else getattr(action, "__dict__", str(action)),
            "raw_reward": raw_reward,
            "reward": external_reward,
            "task_id": task_id,
            "reason_tags": reason_tags
        })

        if raw_reward >= 0.3:
            self._replay_counts[email.get("email_id")] = 0

        self._last_email_id = email.get("email_id")

        _log_step({
            "event": "step",
            "episode_id": self._state.episode_id,
            "email_id": email.get("email_id"),
            "step": self._state.steps,
            "task_id": task_id,
            "action": {
                "category": action.category,
                "priority": action.priority,
                "department": action.department,
                "draft_word_count": len((action.draft_reply or "").split())
            },
            "raw_reward": raw_reward,
            "reward": external_reward,
            "total_reward": self._state.total_reward,
            "done": done,
            "feedback": feedback_items,
            "reason_tags": reason_tags,
            "repetition_ratio": round(repetition_ratio, 4),
            "repetition_penalty": round(rep_penalty, 4),
            "state_features": {
                "multi_intent": email.get("multi_intent"),
                "difficulty": email.get("difficulty"),
                "persona": email.get("persona"),
                "language_variant": email.get("language_variant")
            }
        })

        return self._make_observation(reward=external_reward, raw_reward=raw_reward, done=done, feedback=" | ".join(feedback_items))

    def state(self) -> TriageState:
        if self._state:
            return self._state
        return TriageState(episode_id="not_started", current_task_id=0, total_reward=0.0, steps=0, completed=False)

    def _make_observation(self, reward: float, raw_reward: float, done: bool, feedback: str) -> Dict[str, Any]:
        e = self._current_email or {}
        task_id = self._state.current_task_id if self._state else 0

        feature_hints = {
            "has_money_terms": e.get("has_money_terms", False),
            "has_security_terms": e.get("has_security_terms", False),
            "has_link": e.get("has_link", False),
            "has_urgency": e.get("has_urgency", False),
            "multi_intent": e.get("multi_intent", False),
            "difficulty": e.get("difficulty"),
            "persona": e.get("persona"),
            "language_variant": e.get("language_variant")
        }

        obs_dict: Dict[str, Any] = {
            "email_id": e.get("email_id"),
            "subject": e.get("subject", ""),
            "body": e.get("body", ""),
            "sender": e.get("sender", ""),
            "task_id": task_id,
            "task_description": f"TASK {task_id}: " + ("Classify" if task_id == 1 else "Triage and reply"),
            "reward": reward,
            "done": done,
            "feedback": feedback,
            "feature_hints": feature_hints,
            "episode_id": self._state.episode_id if self._state else None,
            "draft_word_count": None
        }

        obs_dict["raw_reward"] = raw_reward

        return obs_dict

    def export_history(self) -> str:
        return json.dumps(self.history, indent=2, ensure_ascii=False)