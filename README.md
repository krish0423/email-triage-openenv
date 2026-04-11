---
title: Email Triage OpenEnv
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
sdk_version: "5.0.0"
python_version: "3.11"
app_port: 7860
short_description: Confidence-Aware Autonomous Email Routing using Hybrid RL
tags:
  - reinforcement-learning
  - email-triage
  - autonomous-agents
  - nlp
  - fastapi
  - openenv
  - rl-environment
  - llm
fullWidth: true
pinned: false
startup_duration_timeout: 30m
---

A real-world RL environment where an AI agent learns to triage customer support emails — classifying, prioritizing, routing, and drafting professional replies. The agent combines hybrid Q-learning with LLM-backed classification and confidence-gated decision routing across a progressive 3-task curriculum.

## Why This Environment?

Customer email triage is one of the most common yet nuanced tasks in enterprise support. Support agents must:

- Classify emails across overlapping categories (billing vs. refund, technical vs. account)
- Detect disguised phishing attacks hidden among legitimate requests
- Prioritize based on urgency, financial impact, and security risk
- Route to the correct department without unnecessary escalation
- Draft empathetic, context-aware replies that address the customer's specific issue

This environment captures that complexity in a controlled, reproducible setting with multi-dimensional reward signals — making it ideal for training and evaluating AI agents on real-world NLP reasoning.

## Environment Architecture

The environment models a complete customer support pipeline with hybrid RL decision-making:

```
                    ┌──────────────────┐
                    │  Incoming Email  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Feature Extract │
                    │  (NLP + Hints)   │
                    └───┬────┬────┬────┘
                        │    │    │
             ┌──────────┘    │    └──────────┐
             ▼               ▼               ▼
      ┌─────────────┐ ┌────────────┐ ┌─────────────┐
      │  Heuristic   │ │  LLM-based │ │  Q-Learning │
      │  Classifier  │ │  Classifier│ │   Policy    │
      └──────┬───────┘ └─────┬──────┘ └──────┬──────┘
             │               │               │
             └───────┬───────┴───────┬───────┘
                     ▼               │
              ┌──────────────┐       │
              │   Blended    │◄──────┘
              │   Decision   │
              └──────┬───────┘
                     │
           ┌─────────▼──────────┐
           │ Confidence Gating  │
           └──┬───────┬────┬────┘
              │       │    │
              ▼       ▼    ▼
         auto     semi   human
        execute   auto   review
```

Each email passes through feature extraction, three parallel classifiers, a blended decision engine, and confidence-based routing — with category locking for stability.

## Action Space

The agent submits a single structured action per step. Required fields escalate with task difficulty:

| Field          | Type     | Values                                                                   | Required From |
| -------------- | -------- | ------------------------------------------------------------------------ | ------------- |
| `category`     | `string` | `billing`, `technical`, `general`, `complaint`, `refund`, `phishing`     | Task 1        |
| `priority`     | `string` | `P1`, `P2`, `P3`                                                        | Task 2        |
| `department`   | `string` | `billing_team`, `technical_team`, `customer_success`, `returns`, `security` | Task 2     |
| `draft_reply`  | `string` | Professional reply (≥50 words, ≥100 recommended)                        | Task 3        |

Example action:
```json
{
  "category": "billing",
  "priority": "P1",
  "department": "billing_team",
  "draft_reply": null
}
```

## Observation Space

Each observation includes:

- **Email content**: ID, subject, body, and sender address
- **Task context**: Current task ID (1–3) and task description
- **Reward signal**: External reward `(0.01, 0.99)` plus internal `raw_reward [-1.0, 1.0]`
- **Grader feedback**: Human-readable explanation of scoring decisions
- **Feature hints**: Extracted NLP signals to aid classification

```json
{
  "email_id": "email_42",
  "subject": "Urgent: unauthorized charge on my account",
  "body": "I noticed a $149.99 charge I didn't authorize...",
  "sender": "jane.doe@example.com",
  "task_id": 2,
  "task_description": "TASK 2: Triage and reply",
  "reward": 0.6,
  "done": false,
  "feedback": "Correct category | Priority correct",
  "feature_hints": {
    "has_money_terms": true,
    "has_security_terms": false,
    "has_link": false,
    "has_urgency": true,
    "multi_intent": false,
    "difficulty": "medium",
    "persona": "frustrated_customer",
    "language_variant": "formal"
  }
}
```

## Reward Design

Rewards are **multi-dimensional** and provide signal at every step (not just at terminal state). All scores are strictly within `(0.01, 0.99)`:

| Dimension                | Weight | Description                                                        |
| ------------------------ | ------ | ------------------------------------------------------------------ |
| Category accuracy        | 0.40   | Correct classification into one of 6 categories                    |
| Phishing detection       | 0.50   | Bonus for detecting disguised phishing; heavy penalty for misses   |
| Priority correctness     | 0.20   | Correct urgency level assignment (Tasks 2–3)                       |
| Department routing       | 0.20   | Routing to the correct team (Tasks 2–3)                            |
| Reply quality (LLM)      | 0.40   | LLM-judged empathy, relevance, and professionalism (Task 3)        |
| Repetition penalty       | -0.20  | Deduction for repetitive/filler content in draft replies            |

**Partial progress**: Agents earn incremental rewards for each correct sub-decision — not just for final resolution.

**Penalties**: Invalid actions receive `-0.5`. Missed phishing attacks receive `-0.5`. Short or repetitive replies are penalized proportionally.

### Routing Intelligence

The confidence-gated routing determines autonomy level:

| Confidence Level | Routing Action    | Description                                     |
| :--------------- | :---------------- | :---------------------------------------------- |
| **> 0.85**       | 🚀 `auto_execute` | Fully autonomous resolution and reply.          |
| **0.40 – 0.85**  | ⚙️ `semi_auto`    | Drafts prepared, requires quick human sign-off. |
| **≤ 0.40**       | 👨‍💼 `human_review` | Escalated directly to human agents.             |

### Task 1: Email Classification (Easy)

**Scenario**: Classify a customer email into one of 6 categories. Emails range from straightforward billing inquiries to adversarial phishing disguised as legitimate requests.

**Expected approach**: Extract NLP features → Apply heuristic + LLM classification → Submit category

**Max reward per step**: 0.99

### Task 2: Prioritization & Routing (Medium)

**Scenario**: Classify the email, assign a priority level (P1/P2/P3), and route to the correct department. Must balance urgency detection with accurate routing.

**Expected approach**: Classify → Assess urgency signals → Map category to department → Validate with confidence gating

**Max reward per step**: 0.99

### Task 3: Full Triage with Draft Reply (Hard)

**Scenario**: Complete triage including a professional, empathetic draft reply (≥100 words). Replies are scored by an LLM judge on relevance, empathy, and actionability. Phishing emails require security-specific language.

**Expected approach**: Full classification + routing → Generate context-aware reply → Include customer name and issue details → Avoid repetition

**Max reward per step**: 0.99

**Max episode reward: ~2.97** (0.99 per task across the 3-task curriculum)

---

1. **Progressive Difficulty**: 3-task curriculum from classification to full triage with reply generation
2. **Adversarial Phishing**: Disguised phishing emails that test the agent's security awareness
3. **Hybrid RL Engine**: Blended Q-learning + LLM + heuristic classification with confidence weighting
4. **LLM Judge**: LLM-based reply quality scoring for empathy, relevance, and professionalism
5. **Hard-Negative Replay**: Failed emails are automatically re-presented for accelerated learning
6. **Confidence Gating**: Dynamic threshold-based routing (auto/semi-auto/human review)
7. **Multi-dimensional Grading**: Six independent scoring dimensions, not just pass/fail

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)

### Local Development
```bash
# Clone and set up
git clone https://github.com/krish0423/email-triage-openenv.git
cd email-triage-openenv

# Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal, run the baseline agent
export ENV_URL="http://localhost:7860"
export HF_TOKEN="your_groq_api_key"
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
python inference.py
```

### Docker
```bash
# Build the image
docker build -t email-triage-openenv .

# Run
docker run -p 7860:7860 email-triage-openenv

# Test health check
curl http://localhost:7860/health
```

### API Quick Start
```python
import requests

BASE = "http://localhost:7860"

# Reset environment
obs = requests.post(f"{BASE}/reset").json()

# Take an action
result = requests.post(f"{BASE}/step", json={
    "category": "billing",
    "priority": "P1",
    "department": "billing_team",
    "draft_reply": None
}).json()

print(result["reward"])
print(result["feedback"])
print(result["done"])
```

## Baseline Results

Scores from `inference.py` using **llama-3.1-8b-instant** via Groq.
Environment: `https://krishshah07-email-triage-env.hf.space`

| Task                          | Difficulty | Reward Range   | Description                        |
| ----------------------------- | ---------- | -------------- | ---------------------------------- |
| T1 — Email Classification     | Easy       | (0.01, 0.99)   | Category classification            |
| T2 — Prioritization & Routing | Medium     | (0.01, 0.99)   | Category + Priority + Routing      |
| T3 — Full Triage + Reply      | Hard       | (0.01, 0.99)   | Full triage with draft reply       |

### Score Dimensions

| Dimension                | Max   | Description                                              |
| ------------------------ | ----- | -------------------------------------------------------- |
| Category accuracy        | 0.40  | Correct classification                                   |
| Phishing detection       | 0.50  | Detecting disguised phishing attacks                     |
| Priority correctness     | 0.20  | Correct urgency assignment                               |
| Department routing       | 0.20  | Correct team routing                                     |
| Reply quality (LLM)      | 0.40  | Empathy, relevance, professionalism                      |
| Repetition penalty       | -0.20 | Deduction for repetitive reply content                   |

## Environment Variables

| Variable       | Description            | Required                                                     |
| -------------- | ---------------------- | ------------------------------------------------------------ |
| `API_BASE_URL` | LLM API endpoint       | Yes — default: `https://api.groq.com/openai/v1`             |
| `MODEL_NAME`   | Model identifier       | Yes — default: `llama-3.1-8b-instant`                       |
| `HF_TOKEN`     | Groq / API key         | Yes (mandatory, no default)                                  |
| `ENV_URL`      | Environment server URL | No (default: `http://localhost:7860`)                        |
| `PORT`         | Server port            | No (default: 7860)                                           |

## Project Structure

```
.
├── inference.py                         # LLM-powered triage agent
├── openenv.yaml                         # OpenEnv specification
├── models.py                            # Pydantic action/observation/state/reward models
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── .env.example                         # Environment variable template
├── .dockerignore
├── server/
│   ├── app.py                           # FastAPI server (reset/step/state/health)
│   ├── email_triage_environment.py      # Core RL environment logic
│   ├── email_dataset.py                 # Email dataset loader and helper
│   └── llm_judge.py                     # LLM-based reply quality scorer
└── README.md
```
