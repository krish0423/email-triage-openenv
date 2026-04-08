"""
FastAPI server for Email Triage OpenEnv environment.
Endpoints: POST /reset, POST /step, GET /state, GET /health
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
import uvicorn

from models import TriageAction, TriageObservation, TriageState
from server.email_triage_environment import EmailTriageEnvironment

app = FastAPI(
    title="Email Triage OpenEnv",
    description="RL environment where an AI agent learns to triage customer support emails.",
    version="1.0.0",
)

# Initialize environment
env = EmailTriageEnvironment()


# ------------------ ROOT ------------------
@app.get("/")
async def root():
    return {"message": "Email Triage OpenEnv is running 🚀"}


# ------------------ HEALTH ------------------
@app.get("/health")
async def health():
    return {"status": "ok", "env": "email_triage"}


# ------------------ RESET ------------------
@app.post("/reset")
async def reset():
    obs = env.reset()
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
                "task_id": 1,
                "name": "Email Classification",
                "difficulty": "easy",
                "description": "Classify an email into the correct category.",
                "reward_range": [0.0, 1.0],
            },
            {
                "task_id": 2,
                "name": "Prioritization & Routing",
                "difficulty": "medium",
                "description": "Classify, set priority, and route to the correct department.",
                "reward_range": [0.0, 1.0],
            },
            {
                "task_id": 3,
                "name": "Full Triage with Draft Reply",
                "difficulty": "hard",
                "description": "Full triage including a professional draft response to the customer.",
                "reward_range": [0.0, 1.0],
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