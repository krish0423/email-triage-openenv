"""
FastAPI server for Email Triage OpenEnv environment.
Endpoints: POST /reset, POST /step, GET /state, GET /health, GET /metadata, GET /schema, GET /tasks
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from models import TriageAction, TriageObservation, TriageReward, TriageState  # ← TriageReward added
from server.email_triage_environment import EmailTriageEnvironment

app = FastAPI(
    title="Email Triage OpenEnv",
    description="RL environment where an AI agent learns to triage customer support emails.",
    version="1.0.0",
)

env = EmailTriageEnvironment()

TASK_MAP = {
    "task1_email_classification":  1,
    "task2_prioritization_routing": 2,
    "task3_full_triage_reply":      3,
}


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    action: Optional[TriageAction] = None
    # Flat fields accepted for backward compatibility
    category:    Optional[str] = None
    priority:    Optional[str] = None
    department:  Optional[str] = None
    draft_reply: Optional[str] = None


# ------------------ ROOT ------------------
@app.get("/")
async def root():
    return {"message": "Email Triage OpenEnv is running 🚀"}


# ------------------ HEALTH ------------------
@app.get("/health")
async def health():
    return {"status": "ok", "env": "email-triage-env"}


# ------------------ METADATA ------------------
@app.get("/metadata")
async def metadata():
    return {
        "name": "email-triage-env",
        "version": "1.0.0",
        "description": "A real-world RL environment for autonomous customer email triage.",
        "author": "Krish Shah",
        "license": "MIT",
        "tasks": [
            {
                "id": "task1_email_classification",
                "name": "Email Classification",
                "difficulty": "easy",
                "max_steps": 10,
                "description": "Classify an email into the correct category.",
                "score_range": [0.01, 0.99],
                "grader_dimensions": ["classification_accuracy"],
            },
            {
                "id": "task2_prioritization_routing",
                "name": "Prioritization & Routing",
                "difficulty": "medium",
                "max_steps": 10,
                "description": "Classify, set priority, and route to the correct department.",
                "score_range": [0.01, 0.99],
                "grader_dimensions": ["classification_accuracy", "priority_accuracy", "routing_accuracy"],
            },
            {
                "id": "task3_full_triage_reply",
                "name": "Full Triage with Draft Reply",
                "difficulty": "hard",
                "max_steps": 10,
                "description": "Full triage including a professional draft response.",
                "score_range": [0.01, 0.99],
                "grader_dimensions": [
                    "classification_accuracy", "priority_accuracy",
                    "routing_accuracy", "reply_quality",
                ],
            },
        ],
    }


# ------------------ SCHEMA ------------------
@app.get("/schema")
async def schema():
    return {
        "action":      TriageAction.model_json_schema(),
        "observation": TriageObservation.model_json_schema(),
        "reward":      TriageReward.model_json_schema(),   # ← was missing
        "state":       TriageState.model_json_schema(),
    }


# ------------------ RESET ------------------
@app.post("/reset")
async def reset(req: Optional[ResetRequest] = None):
    task_id_str = req.task_id if req else None
    phase       = TASK_MAP.get(task_id_str, 1) if task_id_str else 1
    obs_dict    = env.reset(phase=phase)

    reward_val = max(0.01, min(0.99, float(obs_dict.get("reward", 0.01))))

    return {
        "observation": obs_dict,
        "reward": {
            "value":       reward_val,
            "step_reward": reward_val,
            "cumulative":  reward_val,
            "feedback":    obs_dict.get("feedback", ""),
        },
        "done": obs_dict.get("done", False),
        "info": {},
    }


# ------------------ STEP ------------------
@app.post("/step")
async def step(req: StepRequest):
    try:
        # Build TriageAction from either nested or flat fields
        if req.action:
            action = req.action
        else:
            action = TriageAction(
                category=req.category,
                priority=req.priority,
                department=req.department,
                draft_reply=req.draft_reply,
            )

        obs_dict = env.step(action)

        reward_val = max(0.01, min(0.99, float(obs_dict.get("reward", 0.01))))

        # Guard against env.state() failing before first reset or on error
        try:
            cumulative = max(0.01, min(0.99, float(env.state().total_reward)))
        except Exception:
            cumulative = reward_val

        return {
            "observation": obs_dict,
            "reward": {
                "value":       reward_val,
                "step_reward": reward_val,
                "cumulative":  cumulative,
                "feedback":    obs_dict.get("feedback", ""),
            },
            "done": obs_dict.get("done", False),
            "info": {},
        }

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:                                          # ← broadened
        raise HTTPException(status_code=500, detail=str(e))


# ------------------ STATE ------------------
@app.get("/state")
async def state():
    try:
        s = env.state()
        return s.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State unavailable: {e}")


# ------------------ TASKS ------------------
@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "task_id":     "task1_email_classification",
                "name":        "Email Classification",
                "difficulty":  "easy",
                "description": "Classify an email into the correct category.",
                "reward_range": [0.01, 0.99],
            },
            {
                "task_id":     "task2_prioritization_routing",
                "name":        "Prioritization & Routing",
                "difficulty":  "medium",
                "description": "Classify, set priority, and route to the correct department.",
                "reward_range": [0.01, 0.99],
            },
            {
                "task_id":     "task3_full_triage_reply",
                "name":        "Full Triage with Draft Reply",
                "difficulty":  "hard",
                "description": "Full triage including a professional draft response.",
                "reward_range": [0.01, 0.99],
            },
        ]
    }


# ------------------ MAIN ------------------
def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()