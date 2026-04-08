# 🚀 Email Triage OpenEnv

**Confidence‑Aware Autonomous Email Routing using Hybrid Reinforcement Learning**

Email Triage OpenEnv is a real-world Reinforcement Learning (RL) environment designed to train AI agents for customer email triage. By combining hybrid RL policies with rule-based guardrails and confidence-aware routing, this environment enables the development of autonomous agents capable of intelligent, safe, and highly explainable decision-making.

---

## 💡 Core Innovation

Our agent goes beyond traditional classification systems by:

- **Dynamic Decision Making:** Seamlessly switching between an RL policy and rule-based fallbacks.
- **Confidence-Gated Routing:** Using dynamic confidence thresholds to route emails for auto-resolution, semi-automated handling, or human review.
- **Decision Stability:** Utilizing category locking to maintain consistent state evaluation.
- **Complete Explainability:** Generating human‑readable decision traces for every action taken by the agent.

---

## 🧠 System Architecture

### Hybrid Decision Engine

The core intelligence relies on a multi-layered approach to ensure safety and accuracy:

1. **RL Policy:** Q‑learning based decision making.
2. **Rule‑Based Fallback:** A safety layer to catch edge cases.
3. **Confidence Gating:** Threshold-based action validation.
4. **Category Locking:** Ensures stability across multi-step triage.

### Routing Intelligence

| Confidence Level | Routing Action    | Description                                     |
| :--------------- | :---------------- | :---------------------------------------------- |
| **> 0.85**       | 🚀 `auto_execute` | Fully autonomous resolution and reply.          |
| **0.40 – 0.85**  | ⚙️ `semi_auto`    | Drafts prepared, requires quick human sign-off. |
| **≤ 0.40**       | 👨‍💼 `human_review` | Escalated directly to human agents.             |

---

## 🎯 Multi‑Step Learning Environment

Each episode simulates a realistic triage workflow, escalating in difficulty. **Maximum Episode Reward: 3.0**

| Task   | Description                   | Difficulty | Reward |
| :----- | :---------------------------- | :--------- | :----- |
| **T1** | Category classification       | Easy       | 1.0    |
| **T2** | Category + Priority + Routing | Medium     | 1.0    |
| **T3** | Full triage + Draft reply     | Hard       | 1.0    |

---

## ⚙️ Environment Spaces

### Observation Space

The environment returns the following fields after every `reset()` and `step()`:

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
  "feedback": "string",
  "feature_hints": {
    "has_money_terms": false,
    "has_security_terms": false,
    "has_link": false,
    "has_urgency": false,
    "multi_intent": false,
    "difficulty": "string",
    "persona": "string",
    "language_variant": "string"
  }
}
```

### Action Space

The agent must output a structured decision matching the following schema:

```json
{
  "category": "billing | technical | general | complaint | refund | phishing",
  "priority": "P1 | P2 | P3",
  "department": "billing_team | technical_team | customer_success | returns | security",
  "draft_reply": "string | null"
}
```

---

## 📊 Evaluation Metrics

To effectively track agent performance and safety, the environment measures:

- **📈 Average Episode Reward:** Overall task completion success.
- **⚠️ Confident‑Wrong Rate:** Critical safety metric for incorrect high-confidence actions.
- **🔁 Rule Override Rate:** Frequency of rule-based guardrail activation.
- **🤖 Auto‑Resolve Rate:** Percentage of tickets handled without human intervention.
- **❓ Neutral State Fraction:** Rate of deterministic neutral handling.

---

## 🚀 Quickstart

### 1. Local Setup

Clone the repository and set up your virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the Environment Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 3. Run the Agent

```bash
export ENV_URL="http://localhost:7860"
python inference.py
```

---

## 🔑 API Integration

To run inference with an LLM provider, set these environment variables before running:

```bash
export HF_TOKEN="your_api_key_here"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"

python inference.py
```

---

## 🧪 Example Decision Output

Every decision processed by the agent includes an explainable trace:

```json
{
  "category": "phishing",
  "priority": "P1",
  "department": "security",
  "route": "semi_auto",
  "confidence": 0.52,
  "decision_trace": ["rl_candidate", "hybrid_rule", "locked_category"]
}
```

---

## 📁 Project Structure

```
.
├── inference.py              # Baseline agent — must be in root
├── openenv.yaml              # OpenEnv spec
├── models.py                 # Pydantic action/observation/state models
├── requirements.txt
├── Dockerfile
├── server/
│   ├── app.py                # FastAPI server (reset/step/state endpoints)
│   ├── email_triage_environment.py
│   ├── email_dataset.py
│   └── llm_judge.py
└── README.md
```
