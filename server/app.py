"""
FastAPI server for Email Triage OpenEnv environment.
Endpoints: POST /reset, POST /step, GET /state, GET /health, GET /metadata, GET /schema
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from models import TriageAction, TriageObservation, TriageState
from server.email_triage_environment import EmailTriageEnvironment

app = FastAPI(
    title="Email Triage OpenEnv",
    description="RL environment where an AI agent learns to triage customer support emails.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize environment
env = EmailTriageEnvironment()

# Task ID mapping: string task IDs → internal task phases
TASK_MAP = {
    "task1_email_classification": 1,
    "task2_prioritization_routing": 2,
    "task3_full_triage_reply": 3,
}


# ── Request Models ──────────────────────────────
class ResetRequest(BaseModel):
    task_id: Optional[str] = None


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
        "description": "A real-world RL environment for autonomous customer email triage using hybrid reinforcement learning.",
        "author": "Krish Shah",
        "license": "MIT",
        "tasks": [
            {
                "id": "task1_email_classification",
                "name": "Email Classification",
                "difficulty": "easy",
                "max_steps": 10,
                "description": "Classify an email into the correct category.",
                "score_range": [0.0, 1.0],
                "grader_dimensions": ["classification_accuracy"],
            },
            {
                "id": "task2_prioritization_routing",
                "name": "Prioritization & Routing",
                "difficulty": "medium",
                "max_steps": 10,
                "description": "Classify, set priority, and route to the correct department.",
                "score_range": [0.0, 1.0],
                "grader_dimensions": [
                    "classification_accuracy",
                    "priority_accuracy",
                    "routing_accuracy",
                ],
            },
            {
                "id": "task3_full_triage_reply",
                "name": "Full Triage with Draft Reply",
                "difficulty": "hard",
                "max_steps": 10,
                "description": "Full triage including a professional draft response to the customer.",
                "score_range": [0.0, 1.0],
                "grader_dimensions": [
                    "classification_accuracy",
                    "priority_accuracy",
                    "routing_accuracy",
                    "reply_quality",
                ],
            },
        ],
    }


# ------------------ SCHEMA ------------------
@app.get("/schema")
async def schema():
    return {
        "action": TriageAction.model_json_schema(),
        "observation": TriageObservation.model_json_schema(),
        "state": TriageState.model_json_schema(),
    }


# ------------------ RESET ------------------
@app.post("/reset")
async def reset(req: Optional[ResetRequest] = None):
    task_id_str = req.task_id if req else None
    phase = TASK_MAP.get(task_id_str, 1) if task_id_str else 1
    obs = env.reset(phase=phase)
    return obs


# ------------------ STEP ------------------
@app.post("/step")
async def step(action: TriageAction):
    try:
        obs = env.step(action)
        return obs
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ------------------ STATE ------------------
@app.get("/state")
async def state():
    s = env.state()
    return s.model_dump()


# ------------------ TASKS ------------------
@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "task_id": "task1_email_classification",
                "name": "Email Classification",
                "difficulty": "easy",
                "description": "Classify an email into the correct category.",
                "reward_range": [0.0, 1.0],
                "grader_dimensions": ["classification_accuracy"],
            },
            {
                "task_id": "task2_prioritization_routing",
                "name": "Prioritization & Routing",
                "difficulty": "medium",
                "description": "Classify, set priority, and route to the correct department.",
                "reward_range": [0.0, 1.0],
                "grader_dimensions": [
                    "classification_accuracy",
                    "priority_accuracy",
                    "routing_accuracy",
                ],
            },
            {
                "task_id": "task3_full_triage_reply",
                "name": "Full Triage with Draft Reply",
                "difficulty": "hard",
                "description": "Full triage including a professional draft response to the customer.",
                "reward_range": [0.0, 1.0],
                "grader_dimensions": [
                    "classification_accuracy",
                    "priority_accuracy",
                    "routing_accuracy",
                    "reply_quality",
                ],
            },
        ]
    }


# ------------------ MAIN ------------------
def main():
    """Callable entry point for [project.scripts] and programmatic use."""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()