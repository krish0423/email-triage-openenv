from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any, List


# ── Actions ──────────────────────────────────────────────────────────────────

class TriageAction(BaseModel):
    """
    Action submitted by the agent for each task.

    Task 1 — category only
    Task 2 — category + priority + department
    Task 3 — category + priority + department + draft_reply (>=100 words)

    All enums must exactly match openenv.yaml action_space declarations.
    """

    category: Optional[Literal[
        "billing", "technical", "general", "complaint", "refund", "phishing"
    ]] = None

    priority: Optional[Literal["P1", "P2", "P3"]] = None

    department: Optional[Literal[
        "billing_team", "technical_team", "customer_success", "returns", "security"
    ]] = None

    draft_reply: Optional[str] = None

    model_config = {"from_attributes": True}

    def validate_for_task(self, task_id: int) -> List[str]:
        """
        Lightweight pre-send validation. Returns a list of error strings
        (empty list means action is valid for this task).
        """
        errors: List[str] = []
        if task_id >= 1 and self.category is None:
            errors.append("category is required for task >= 1")
        if task_id >= 2:
            if self.priority is None:
                errors.append("priority is required for task >= 2")
            if self.department is None:
                errors.append("department is required for task >= 2")
        if task_id == 3:
            if not self.draft_reply or len(self.draft_reply.split()) < 100:
                errors.append("draft_reply must be >= 100 words for task 3")
        return errors


# ── Observations ─────────────────────────────────────────────────────────────

class TriageObservation(BaseModel):
    """
    What the agent sees after /reset or /step.

    task_id is a numeric int (1, 2, 3) matching the server's internal
    phase representation. Inference code maps string task IDs to these
    integers via TASK_PHASE.
    """

    email_id:         str
    subject:          str
    body:             str
    sender:           str
    task_id:          int           # 1, 2, or 3 — numeric, not the string slug
    task_description: str
    max_steps:        int   = 10    # must match openenv.yaml per-task max_steps (all tasks = 10)
    reward:           float = 0.01  # clamped minimum — never 0.0 (evaluator rejects exact 0)
    done:             bool  = False
    feedback:         str   = ""    # grader feedback for agent to learn from

    # Optional enrichment from the email dataset
    feature_hints:    Optional[Dict[str, Any]] = None
    persona:          Optional[str]             = None
    language_variant: Optional[str]             = None
    difficulty:       Optional[str]             = None

    model_config = {"from_attributes": True}


# ── Reward ───────────────────────────────────────────────────────────────────

class TriageRewardBreakdown(BaseModel):
    """
    Per-dimension scores matching openenv.yaml grader dimensions.
    Unused dimensions default to 0.01 (floor) for tasks that don't score them.

    Task 1: only classification_accuracy is active
    Task 2: classification_accuracy + priority_accuracy + routing_accuracy
    Task 3: all four dimensions
    """

    classification_accuracy: float = Field(default=0.01, ge=0.01, le=0.99)
    priority_accuracy:        float = Field(default=0.01, ge=0.01, le=0.99)
    routing_accuracy:         float = Field(default=0.01, ge=0.01, le=0.99)
    reply_quality:            float = Field(default=0.01, ge=0.01, le=0.99)

    model_config = {"from_attributes": True}


class TriageReward(BaseModel):
    """
    Reward signal returned by /step.
    value is strictly within (0.01, 0.99) — evaluator rejects 0.0 and 1.0.
    """

    value:      float                = Field(default=0.01, ge=0.01, le=0.99)
    step_reward: float               = Field(default=0.01, ge=0.01, le=0.99)
    cumulative:  float               = Field(default=0.01, ge=0.01, le=0.99)
    breakdown:  TriageRewardBreakdown = Field(default_factory=TriageRewardBreakdown)
    feedback:   str                  = Field(default="")

    model_config = {"from_attributes": True}


# ── State ────────────────────────────────────────────────────────────────────

class TriageState(BaseModel):
    """
    Internal episode state returned by GET /state.
    total_reward accumulates across steps within an episode.
    """

    episode_id:      str
    current_task_id: int   = 1     # numeric (1, 2, 3)
    total_reward:    float = 0.01  # clamped — never 0.0
    steps:           int   = 0
    completed:       bool  = False

    model_config = {"from_attributes": True}