from pydantic import BaseModel, validator
from typing import Optional, Literal, Dict, Any, List


# ── Actions ──────────────────────────────────────────────────────────────────

class TriageAction(BaseModel):
    """Action submitted by the agent for any task."""
    # Task 1 – classify (phishing added as a security-aware category)
    category: Optional[Literal["billing", "technical", "general", "complaint", "refund", "phishing", "account"]] = None

    # Task 2 – classify + prioritize + route
    priority: Optional[Literal["P1", "P2", "P3"]] = None
    department: Optional[Literal[
    "billing_team", "technical_team", "customer_success", "returns", "security"
]] = None


    # Task 3 – full triage with drafted reply
    draft_reply: Optional[str] = None

    class Config:
        orm_mode = True

    def validate_for_task(self, task_id: int) -> List[str]:
        """
        Lightweight validation helper the trainer/agent can call before sending action.
        Returns a list of error messages (empty if valid).
        """
        errors: List[str] = []
        if task_id >= 1:
            if self.category is None:
                errors.append("category is required for task >= 1")
        if task_id >= 2:
            if self.priority is None:
                errors.append("priority is required for task >= 2")
            if self.department is None:
                errors.append("department is required for task >= 2")
        if task_id == 3:
         if not self.draft_reply or len(self.draft_reply.split()) < 50:
           errors.append("draft_reply must be >= 50 words for task 3")

        return errors


# ── Observations ─────────────────────────────────────────────────────────────

class TriageObservation(BaseModel):
    """What the agent sees after reset() or step()."""
    email_id: str
    subject: str
    body: str
    sender: str
    task_id: int          # 1, 2, or 3
    task_description: str
    reward: float = 0.0
    done: bool = False
    feedback: str = ""    # Grader feedback for the agent

    # Optional hints / metadata from dataset/environment for explainability and features
    feature_hints: Optional[Dict[str, Any]] = None
    # convenience top-level fields (also available inside feature_hints)
    persona: Optional[str] = None
    language_variant: Optional[str] = None
    difficulty: Optional[str] = None

    class Config:
        orm_mode = True


# ── State ─────────────────────────────────────────────────────────────────────

class TriageState(BaseModel):
    """Internal episode state."""
    episode_id: str
    current_task_id: int = 1
    total_reward: float = 0.0
    steps: int = 0
    completed: bool = False

    class Config:
        orm_mode = True
