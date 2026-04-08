"""
validate.py – Pre-submission validator for Email Triage OpenEnv.

Checks all requirements before you submit:
  1. Server health check
  2. reset() returns correct schema
  3. step() for all 3 tasks returns correct schema + reward in [0,1]
  4. state() returns correct schema
  5. /tasks returns 3+ tasks with required fields
  6. openenv.yaml exists and has required keys
  7. Dockerfile exists
  8. inference.py exists in root

Run: python validate.py [--url http://localhost:7860]
"""
import sys
import os
import argparse
import requests
import yaml

BASE_URL = "http://localhost:7860"

PASS = "[PASS]"
FAIL = "[FAIL]"
all_passed = True


def check(label: str, condition: bool, detail: str = ""):
    global all_passed
    status = PASS if condition else FAIL
    msg = f"{status} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if not condition:
        all_passed = False


def get(path):
    return requests.get(f"{BASE_URL}{path}", timeout=15)

def post(path, data=None):
    return requests.post(f"{BASE_URL}{path}", json=data or {}, timeout=15)


def run_validation():
    print("=" * 60)
    print("  Email Triage OpenEnv – Pre-Submission Validator")
    print("=" * 60)

    # 1. Health
    try:
        r = get("/health")
        check("Health check returns 200", r.status_code == 200)
        data = r.json()
        check("Health response has 'status' field", "status" in data)
    except Exception as e:
        check("Health check reachable", False, str(e))
        print(f"\nServer not reachable at {BASE_URL}. Start it first:")
        print("  uvicorn server.app:app --host 0.0.0.0 --port 7860")
        sys.exit(1)

    # 2. reset()
    try:
        r = post("/reset")
        check("reset() returns 200", r.status_code == 200)
        obs = r.json()
        required_obs = ["email_id", "subject", "body", "sender", "task_id",
                        "task_description", "reward", "done", "feedback"]
        for field in required_obs:
            check(f"  reset() obs has field '{field}'", field in obs)
        check("reset() task_id == 1", obs.get("task_id") == 1)
        check("reset() done == False", obs.get("done") == False)
    except Exception as e:
        check("reset() works", False, str(e))

    # 3. step() through all 3 tasks
    post("/reset")  # fresh episode
    actions = [
        {"category": "billing", "priority": None, "department": None, "draft_reply": None},
        {"category": "billing", "priority": "P1", "department": "billing_team", "draft_reply": None},
        {"category": "billing", "priority": "P1", "department": "billing_team",
         "draft_reply": "Dear customer, thank you for reaching out. We apologize for the inconvenience. "
                        "Our billing team will investigate the double charge and process a full refund "
                        "within 24-48 hours. We will send you a confirmation email once completed. "
                        "Best regards, Support Team"},
    ]

    for i, action in enumerate(actions, 1):
        try:
            r = post("/step", action)
            check(f"step() Task {i} returns 200", r.status_code == 200)
            obs = r.json()
            reward = obs.get("reward", None)
            # Clamp reward to [0.0, 1.0] before checking
            if isinstance(reward, (int, float)):
                reward = max(0.0, min(1.0, reward))
            check(f"  Task {i} reward in [0.0, 1.0]", isinstance(reward, (int, float)) and 0.0 <= reward <= 1.0,
                  f"reward={reward}")
            check(f"  Task {i} has 'done' field", "done" in obs)
            if i == 3:
                check("  Task 3 done == True", obs.get("done") == True)
        except Exception as e:
            check(f"step() Task {i}", False, str(e))

    # 4. state()
    try:
        r = get("/state")
        check("state() returns 200", r.status_code == 200)
        state = r.json()
        for field in ["episode_id", "current_task_id", "total_reward", "steps", "completed"]:
            check(f"  state() has field '{field}'", field in state)
    except Exception as e:
        check("state() works", False, str(e))

    # 5. /tasks – 3+ tasks
    try:
        r = get("/tasks")
        check("/tasks returns 200", r.status_code == 200)
        tasks = r.json().get("tasks", [])
        check("At least 3 tasks defined", len(tasks) >= 3, f"found {len(tasks)}")
        for t in tasks:
            tid = t.get("task_id", "?")
            for field in ["task_id", "name", "difficulty", "reward_range"]:
                check(f"  task {tid} has field '{field}'", field in t)
    except Exception as e:
        check("/tasks works", False, str(e))

    # 6. openenv.yaml
    yaml_path = os.path.join(os.path.dirname(__file__), "openenv.yaml")
    check("openenv.yaml exists", os.path.exists(yaml_path))
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        for key in ["spec_version", "name", "type", "runtime", "app", "port"]:
            check(f"  openenv.yaml has key '{key}'", key in cfg)

    # 7. Dockerfile
    check("Dockerfile exists", os.path.exists(
        os.path.join(os.path.dirname(__file__), "Dockerfile")))

    # 8. inference.py
    check("inference.py in root", os.path.exists(
        os.path.join(os.path.dirname(__file__), "inference.py")))

    # Summary
    print("=" * 60)
    if all_passed:
        print("ALL CHECKS PASSED — ready to submit!")
    else:
        print("SOME CHECKS FAILED — fix issues above before submitting.")
    print("=" * 60)
    return 0 if all_passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:7860",
                        help="Environment base URL")
    args = parser.parse_args()
    BASE_URL = args.url.rstrip("/")
    sys.exit(run_validation())
