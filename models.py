from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any, List


# ── Actions ──────────────────────────────────────────────────────────────────

class TriageAction(BaseModel):
    """Action submitted by the agent for any task."""

    # Task 1 — category only
    # NOTE: must match openenv.yaml action_space enum exactly — no "account"
    category: Optional[Literal[
        "billing", "technical", "general", "complaint", "refund", "phishing"
    ]] = None

    # Task 2 — add priority + department
    priority: Optional[Literal["P1", "P2", "P3"]] = None
    department: Optional[Literal[
        "billing_team", "technical_team", "customer_success", "returns", "security"
    ]] = None

    # Task 3 — add draft reply
    draft_reply: Optional[str] = None

    model_config = {"from_attributes": True}  # replaces class Config in Pydantic v2

    def validate_for_task(self, task_id: int) -> List[str]:
        errors: List[str] = []
        if task_id >= 1 and self.category is None:
            errors.append("category is required for task >= 1")
        if task_id >= 2:
            if self.priority is None:
                errors.append("priority is required for task >= 2")
            if self.department is None:
                errors.append("department is required for task >= 2")
        if task_id == 3:
            # env.py enforces a 100-word minimum for draft_reply
            if not self.draft_reply or len(self.draft_reply.split()) < 100:
                errors.append("draft_reply must be >= 100 words for task 3")
        return errors


# ── Observations ─────────────────────────────────────────────────────────────

class TriageObservation(BaseModel):
    """What the agent sees after reset() or step()."""
    email_id:         str
    subject:          str
    body:             str
    sender:           str
    # numeric task id to match env.py (1, 2, 3)
    task_id:          int   # numeric task id e.g. 1, 2, 3
    task_description: str
    max_steps:        int   = 25     # per-task step limit; inference.py reads this
    reward:           float = 0.01   # default is clamped minimum, never 0.0
    done:             bool  = False
    feedback:         str   = ""

    feature_hints:    Optional[Dict[str, Any]] = None
    persona:          Optional[str] = None
    language_variant: Optional[str] = None
    difficulty:       Optional[str] = None

    model_config = {"from_attributes": True}


# ── Reward ───────────────────────────────────────────────────────────────────

class TriageRewardBreakdown(BaseModel):
    """Per-dimension reward scores matching openenv.yaml grader dimensions."""
    classification_accuracy: float = Field(default=0.01, ge=0.01, le=0.99)
    priority_accuracy:       float = Field(default=0.01, ge=0.01, le=0.99)
    routing_accuracy:        float = Field(default=0.01, ge=0.01, le=0.99)
    reply_quality:           float = Field(default=0.01, ge=0.01, le=0.99)


class TriageReward(BaseModel):
    """Reward signal returned by the grader. Value is strictly in (0.01, 0.99)."""
    value:     float = Field(default=0.01, ge=0.01, le=0.99)
    breakdown: TriageRewardBreakdown = Field(default_factory=TriageRewardBreakdown)
    feedback:  str   = Field(default="")

    model_config = {"from_attributes": True}


# ── State ────────────────────────────────────────────────────────────────────

class TriageState(BaseModel):
    """Internal episode state returned by GET /state."""
    episode_id:      str
    # numeric current_task_id to match env.py internal representation
    current_task_id: int   = 1
    total_reward:    float = 0.01
    steps:           int   = 0
    completed:       bool  = False

    model_config = {"from_attributes": True}
