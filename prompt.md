# 🧠 Session Resume Prompt — Email Triage OpenEnv (Meta x Scaler Hackathon)

Paste this entire file as your first message in a new Claude session to resume exactly where you left off.

---

## Context

I am participating in the **OpenEnv Meta x Scaler Hackathon (Round 1)**.
- **Deadline**: April 8, 2026, 11:59 PM IST
- **Goal**: Build a complete OpenEnv RL environment, deploy to Hugging Face Spaces, submit URL
- **Framework**: OpenEnv by Meta-PyTorch — Gymnasium-style `step()` / `reset()` / `state()` API served over HTTP via FastAPI + Docker

## What We Have Already Built (v2 — Complete)

The project is called **Email Triage OpenEnv**. An AI agent learns to triage customer support emails across 3 tasks of increasing difficulty per episode.

### File Structure
```
email_triage_env/
├── models.py                          # Pydantic Action / Observation / State
├── openenv.yaml                       # OpenEnv spec file (spec_version: 1)
├── requirements.txt                   # fastapi, uvicorn, pydantic, openai, httpx, pyyaml
├── Dockerfile                         # FROM python:3.11-slim, EXPOSE 7860
├── README.md                          # HF Spaces header + full docs
├── inference.py                       # Baseline script — [START][STEP][END] log format
├── validate.py                        # Pre-submission validator (all checks)
├── client.py                          # Async + sync OpenEnv HTTP client
└── server/
    ├── __init__.py
    ├── app.py                         # FastAPI: /reset /step /state /health /tasks
    ├── email_dataset.py               # 15 real-world emails across 5 categories
    ├── email_triage_environment.py    # Core logic + all 3 graders
    └── llm_judge.py                   # LLM-as-judge for Task 3 (heuristic fallback)
```

### Tasks
| Task | Difficulty | What agent must do | Reward |
|------|-----------|-------------------|--------|
| T1 | Easy | Classify email into correct category | 0.0 or 1.0 (binary) |
| T2 | Medium | Classify + set priority P1/P2/P3 + route to department | 0.0–1.0 partial (1/3 per field) |
| T3 | Hard | Full triage + professional draft reply to customer | 0.0–1.0 (0.2 per field + 0.4 LLM judge on reply) |

### Action Space
```json
{
  "category":    "billing | technical | general | complaint | refund",
  "priority":    "P1 | P2 | P3",
  "department":  "billing_team | tech_support | customer_success | returns",
  "draft_reply": "string (Task 3 only, else null)"
}
```

### Observation Space
```json
{
  "email_id": "string",
  "subject": "string",
  "body": "string",
  "sender": "string",
  "task_id": 1,
  "task_description": "string",
  "reward": 0.0,
  "done": false,
  "feedback": "string"
}
```

### Dataset
15 emails across 5 categories: billing (3), technical (3), general (2), complaint (3), refund (4).
Each email has: `email_id`, `subject`, `body`, `sender`, `ground_truth` (category, priority, department, ideal_reply_keywords).

### Grader Logic
- **T1**: `reward = 1.0` if `action.category == ground_truth.category` else `0.0`
- **T2**: `reward = sum(1/3 for field in [category, priority, department] if correct)`
- **T3**: `reward = sum(0.2 for field in [category, priority, department] if correct) + llm_judge_score(draft_reply) * 0.4`
  - `llm_judge_score` calls the LLM API with a 4-criteria rubric (empathy, accuracy, next step, professionalism)
  - Falls back to `_heuristic_score` if API unavailable (checks length, empathy words, action words, greeting, closing)

### Environment Variables Required
| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint e.g. `https://api.openai.com/v1` |
| `MODEL_NAME` | Model identifier e.g. `gpt-4o-mini` |
| `HF_TOKEN` | Hugging Face / API key |
| `ENV_URL` | Deployed environment URL (default: `http://localhost:7860`) |

### Key Implementation Details
- `models.py` uses **Pydantic v2** (`BaseModel`, not dataclasses)
- Server is **single-session** (`SUPPORTS_CONCURRENT_SESSIONS=False`, `max_concurrent_envs=1`)
- `openenv.yaml` has `spec_version: 1`, `type: step-reset`, `runtime.type: docker`, `port: 7860`
- `inference.py` uses **OpenAI client** for all LLM calls (required by hackathon rules)
- Logs emit strict `[START]`, `[STEP]`, `[END]` JSON lines to stdout — do NOT change this format
- `inference.py` is in the **root directory** (required by hackathon rules)
- Dockerfile: `FROM python:3.11-slim`, runs `uvicorn server.app:app --host 0.0.0.0 --port 7860`

### API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Returns `{"status": "ok"}` |
| POST | `/reset` | Start new episode → returns Observation |
| POST | `/step` | Submit TriageAction → returns Observation with reward |
| GET | `/state` | Returns TriageState (episode_id, task_id, total_reward, steps, completed) |
| GET | `/tasks` | Lists all 3 tasks with metadata |

### Local Run Commands
```bash
# Install
pip install -r requirements.txt

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Validate (separate terminal)
python validate.py --url http://localhost:7860

# Run inference
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_key_here
export ENV_URL=http://localhost:7860
python inference.py

# Docker
docker build -t email-triage-env .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=your_key_here \
  email-triage-env
```

### HF Spaces Deploy
```bash
# Create a new Space at huggingface.co (SDK: Docker)
git init
git add .
git commit -m "Email Triage OpenEnv v2"
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/email-triage-env
git push space main

# Set these as Space Secrets (Settings → Variables and secrets):
# API_BASE_URL, MODEL_NAME, HF_TOKEN
```

### Submission Checklist Status
- [x] Real-world task (not a game/toy)
- [x] `step()` / `reset()` / `state()` implemented
- [x] `openenv.yaml` with full spec
- [x] 3 tasks: easy → medium → hard
- [x] Rewards in 0.0–1.0 range with partial progress
- [x] `inference.py` in root with `[START][STEP][END]` log format
- [x] OpenAI client used for all LLM calls
- [x] `Dockerfile` builds and runs
- [x] `README.md` with action/obs space + setup instructions
- [x] `validate.py` pre-submission checker
- [x] `client.py` async + sync HTTP client
- [ ] Deploy to HF Spaces → submit URL

---

## What Might Still Need Work (continue from here)

Things we might want to improve or haven't done yet:
1. **Chain-of-thought inference prompt** — richer prompting in `inference.py` to boost baseline scores
2. **Multi-session support** — add session ID to enable concurrent episodes
3. **More emails** — expand dataset beyond 15
4. **Task 3 LLM judge tuning** — adjust rubric scoring weights
5. **Any bug fixes** found during local testing
6. **HF Spaces deployment issues** — Dockerfile or dependency problems

---

## How to Ask Claude to Continue

After pasting this file, you can say things like:
- "Continue where we left off — help me deploy to HF Spaces"
- "Fix this error I'm getting: [paste error]"
- "Improve the inference.py with chain-of-thought prompting"
- "I ran validate.py and this check failed: [paste output]"
- "Add multi-session support"
- "Help me write a better README"
