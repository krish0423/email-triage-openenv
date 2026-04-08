# mock_env.py
from flask import Flask, request, jsonify
import random
import uuid
import time

app = Flask(__name__)

# small synthetic dataset for testing
EMAIL_DATASET = [
    {
        "email_id": "e1",
        "subject": "Invoice charge failed",
        "body": "We saw a charge on your card. Please check your invoice.",
        "true_category": "billing",
        "true_priority": "P1",
        "true_department": "billing_team",
        "needs_draft": True,
        "disguised": False
    },
    {
        "email_id": "e2",
        "subject": "Login link - verify now",
        "body": "Click http://fake to verify password immediately",
        "true_category": "phishing",
        "true_priority": "P1",
        "true_department": "tech_support",
        "needs_draft": True,
        "disguised": True
    },
    {
        "email_id": "e3",
        "subject": "App crashes on start",
        "body": "App shows error 500 after update",
        "true_category": "technical",
        "true_priority": "P2",
        "true_department": "tech_support",
        "needs_draft": True,
        "disguised": False
    }
]

# simple server state
SERVER_STATE = {
    "current": None,
    "episode_id": None,
    "last_action": None,
    "episode_result": {}
}

def _pick_email():
    return random.choice(EMAIL_DATASET)

@app.route("/reset", methods=["POST"])
def reset():
    e = _pick_email()
    SERVER_STATE["current"] = e
    SERVER_STATE["episode_id"] = str(uuid.uuid4())
    SERVER_STATE["episode_result"] = {}
    obs = {
        "email_id": e["email_id"],
        "subject": e["subject"],
        "body": e["body"],
        "task_id": 1,
        "reward": 0.0,
        "done": False
    }
    return jsonify(obs)

@app.route("/step", methods=["POST"])
def step():
    action = request.get_json() or {}
    e = SERVER_STATE["current"]
    # simple scoring: category match -> reward 0.5, else 0.0; if phishing disguised and missed -> -0.5
    reward = 0.0
    if action.get("category") == e["true_category"]:
        reward += 0.5
    else:
        reward -= 0.2
    if e.get("disguised"):
        if action.get("category") == "phishing":
            reward += 0.5
        else:
            reward -= 0.5
    # progress task id
    task_id = request.json.get("task_id", 1) if request.json else 1
    # simulate done after 3 tasks
    done = False
    if task_id >= 3:
        done = True
        SERVER_STATE["episode_result"] = {"pipeline_correct": action.get("category") == e["true_category"]}
    # return next obs (advance task id)
    next_obs = {
        "email_id": e["email_id"],
        "subject": e["subject"],
        "body": e["body"],
        "task_id": min(3, task_id + 1),
        "reward": reward,
        "done": done,
        "true_category": e["true_category"],
        "true_priority": e["true_priority"],
        "true_department": e["true_department"],
        "needs_draft": e["needs_draft"]
    }
    return jsonify(next_obs)

@app.route("/state", methods=["GET"])
def state():
    return jsonify({
        "episode_id": SERVER_STATE["episode_id"],
        "episode_result": SERVER_STATE.get("episode_result", {})
    })

if __name__ == "__main__":
    app.run(port=7860)
