# email_dataset.py
# Advanced synthetic email dataset generator with distribution shift, personas,
# adversarial cases, label noise, thread history, intent hierarchy, difficulty scoring,
# language variants, reward-aware expected actions, and edge cases.

import json
import random
import os
import copy
from typing import List, Dict, Optional, Tuple

OUT_JSON = "email_dataset.json"
CATEGORIES = ["phishing", "billing", "technical", "refund", "complaint", "general"]

# generation hyperparameters
DEFAULT_PER_CATEGORY = 50
DEFAULT_CONFUSING = 40
DEFAULT_ADVERSARIAL = 30
DEFAULT_NEGATIVE = 30
LABEL_NOISE_PROB = 0.03  # default 3%

# keyword sets
MONEY_TERMS = ["refund", "money", "invoice", "charge", "payment", "billing"]
SECURITY_TERMS = ["password", "verify", "otp", "credentials", "suspend", "login", "link"]
LINK_TERMS = ["http", "https", "link", "click here"]
URGENCY_TERMS = ["urgent", "asap", "immediately", "within 24 hours"]

HINGLISH_VARIANTS = [
    "pls help asap yaar",
    "mera account block ho gaya",
    "refund kab milega?",
    "app kaam nahi kar raha",
    "kripya madad karein",
]

LANG_VARIANTS = ["pure_english", "hinglish", "formal_english", "broken_english"]

# --- text utilities ---
def inject_typos(s: str, prob: float = 0.03) -> str:
    chars = list(s)
    for i in range(len(chars)):
        if random.random() < prob and len(chars) > 1:
            j = random.randint(0, len(chars) - 1)
            chars[i], chars[j] = chars[j], chars[i]
    return "".join(chars)

def random_case(s: str, prob: float = 0.12) -> str:
    return "".join(ch.upper() if random.random() < prob else ch for ch in s)

def casualize(s: str) -> str:
    if random.random() < 0.25:
        s = s.replace("please", "pls").replace("you", "u")
        s = s.replace("I am", "I'm").replace("do not", "don't")
    return s

def add_exclamations(s: str) -> str:
    if random.random() < 0.25:
        return s + "!!!"
    return s

def add_signature_variants(body: str) -> str:
    if random.random() < 0.45:
        sig_templates = [
            "Regards,\nAlex Smith\nSenior Manager\n{} \n{}",
            "Thanks,\nPriya Patel\nCustomer Success\n{} \n{}",
            "Best,\nRahul Verma\nSupport Lead\n{} \n{}",
            "Sincerely,\nAnita Rao\nBilling Team\n{} \n{}",
        ]
        tmpl = random.choice(sig_templates)
        comp = random.choice(["XYZ Corp", "Acme Ltd", "FinTech Co", "ShopNow", "Global Services"])
        phone = f"+91-{random.randint(7000000000,9999999999)}"
        return body + "\n\n" + tmpl.format(comp, phone)
    return body

def add_email_thread(body: str) -> str:
    if random.random() < 0.3:
        forwarded = (
            "\n\n--- Forwarded message ---\nFrom: support@xyz.com\nSubject: Re: Issue\n\n"
            + body[:200]
        )
        return body + forwarded
    return body

def add_hinglish(body: str) -> str:
    if random.random() < 0.2:
        return body + " " + random.choice(HINGLISH_VARIANTS)
    return body

def make_long_email(body: str) -> str:
    if random.random() < 0.3:
        return body + "\n\n" + " ".join([body] * random.randint(2, 4))
    return body

def noisy_variant(s: str) -> str:
    s = inject_typos(s, prob=0.03)
    s = random_case(s, prob=0.12)
    s = casualize(s)
    s = add_exclamations(s)
    return s

# --- detectors and multi-intent ---
def detect_has_money_terms(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in MONEY_TERMS)

def detect_has_security_terms(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in SECURITY_TERMS)

def detect_has_link(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in LINK_TERMS)

def detect_has_urgency(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in URGENCY_TERMS)

def detect_multi_intent(text: str) -> bool:
    signals = 0
    t = text.lower()
    if any(w in t for w in ["charge", "invoice", "payment"]): signals += 1
    if any(w in t for w in ["error", "bug", "crash", "not working", "can't log"]): signals += 1
    if any(w in t for w in ["refund", "return", "money back"]): signals += 1
    return signals > 1

# --- persona styling ---
PERSONAS = ["non_tech_user", "developer", "angry_customer", "elderly_user"]
def apply_persona_style(text: str, persona: str) -> str:
    if persona == "non_tech_user":
        text = text.replace("error", "not working").replace("crash", "stops working")
    elif persona == "developer":
        text = text + " Stack trace: NullPointerException at line 42."
    elif persona == "angry_customer":
        text = text + " This is unacceptable and I want a refund now!!!"
    elif persona == "elderly_user":
        text = "Hello, " + text + " I am not very technical, please help."
    # small chance to add persona-specific slang
    if random.random() < 0.2:
        text = text + " " + random.choice(["Thanks", "Regards", "Please help"])
    return text

# --- seeds expanded ---
SEEDS = {
    "phishing": [
        ("Suspicious login link", "I received an email asking me to verify my account via a link. It looks suspicious."),
        ("Fake login attempt", "There was a login attempt and the email asks for my password and verification link."),
        ("Verify your account now", "Please verify your account by clicking the link. It asks for my credentials."),
        ("Unrecognized sign-in", "I got an email about a sign-in I don't recognize with a link to reset password."),
        ("Account suspension warning", "Your account will be suspended unless you verify now via the link."),
        ("OTP request", "I received an OTP request I didn't initiate. The email asks for the code."),
        ("Gift card scam", "You won a gift card. Click the link to claim and enter payment details."),
        ("Fake support email", "Support asked me to share credentials to fix my account via a link."),
        ("Urgent account suspension", "Immediate action required: verify account or it will be suspended."),
        ("Credential harvest", "The email asks for bank details and password to 'confirm' identity.")
    ],
    "billing": [
        ("Invoice charge question", "I see a charge on my card I don't recognize. Please explain the invoice."),
        ("Subscription renewal", "My subscription was renewed and charged; I didn't authorize this payment."),
        ("Unexpected charge", "I was billed twice for the same invoice. Please check my account."),
        ("Billing discrepancy", "The invoice amount is different than expected; need clarification."),
        ("Charge on card", "There is a charge on my card that I need help with.")
    ],
    "technical": [
        ("App crashes on launch", "The app crashes every time I open it after the latest update."),
        ("Error 500 on login", "I get Error 500 when trying to log in; site not working."),
        ("Feature broken", "A dashboard feature is broken and throws an exception."),
        ("System not responding", "The system freezes and I cannot complete tasks."),
        ("Bug in mobile app", "Mobile app crashes on startup and shows a stack trace.")
    ],
    "refund": [
        ("Refund not received", "I requested a refund 10 days ago but haven't received it yet."),
        ("Return request", "I returned the item but the refund hasn't been processed."),
        ("Money back not issued", "I was promised a refund and it's not in my account."),
        ("Refund status", "What's the status of my refund request?"),
        ("Request refund", "Please process my refund for the canceled order.")
    ],
    "complaint": [
        ("Very unhappy with service", "I'm extremely disappointed with the service I received."),
        ("Poor experience", "This was the worst experience; I want to file a complaint."),
        ("Unacceptable delay", "My issue was ignored and the delay is unacceptable."),
        ("Rude support", "Support was rude and didn't resolve my problem."),
        ("Service failure", "I expected better; this is unacceptable.")
    ],
    "general": [
        ("Product question", "Can you tell me more about the product features and pricing?"),
        ("How to use feature", "How do I enable the new feature in my account?"),
        ("Request information", "I need documentation or a guide for setup."),
        ("General inquiry", "I have a question about your service and how it works."),
        ("Availability question", "Is this feature available in my region?")
    ]
}

# --- generation functions ---
def generate_examples_per_category(category: str, n_per_category: int = DEFAULT_PER_CATEGORY) -> List[Dict]:
    seeds = SEEDS[category]
    examples = []
    while len(examples) < n_per_category:
        subj, body = random.choice(seeds)
        if random.random() < 0.4:
            subj = subj + " - please help"
        if random.random() < 0.3:
            body = body + " Could you assist?"
        subj_v = noisy_variant(subj)
        body_v = noisy_variant(body)
        # persona
        persona = random.choice(PERSONAS)
        body_v = apply_persona_style(body_v, persona)
        # augmentations
        body_v = add_time_pressure(body_v) if 'add_time_pressure' in globals() else body_v
        body_v = add_signature_variants(body_v)
        body_v = add_email_thread(body_v)
        body_v = add_hinglish(body_v)
        body_v = make_long_email(body_v)
        # language variant tag
        lang = random.choice(LANG_VARIANTS)
        if lang == "hinglish":
            body_v = add_hinglish(body_v)
        text = subj_v + " " + body_v
        # heuristics for priority & department
        if category == "phishing":
            priority = "P1"; dept = "tech_support"
        elif category == "billing":
            priority = "P1"; dept = "billing_team"
        elif category == "technical":
            priority = "P2"; dept = "tech_support"
        elif category == "refund":
            priority = "P2"; dept = "returns"
        elif category == "complaint":
            priority = "P3"; dept = "customer_success"
        else:
            priority = "P3"; dept = "customer_success"
        needs_draft = category in ("phishing", "refund", "complaint", "technical")
        # intent hierarchy: primary = category, secondary heuristics
        intent_primary = category
        intent_secondary = None
        if detect_multi_intent(text):
            # choose a plausible secondary intent
            if category != "billing" and detect_has_money_terms(text):
                intent_secondary = "billing"
            elif category != "technical" and detect_has_security_terms(text):
                intent_secondary = "technical"
            else:
                intent_secondary = random.choice([c for c in CATEGORIES if c != category])
        # difficulty score
        difficulty_score = (
            0.3 * int(detect_multi_intent(text)) +
            0.2 * int(detect_has_link(text)) +
            0.2 * int(detect_has_urgency(text)) +
            0.3 * (min(len(body_v), 400) / 400)
        )
        if difficulty_score < 0.3:
            difficulty = "easy"
        elif difficulty_score < 0.6:
            difficulty = "medium"
        else:
            difficulty = "hard"
        example = {
            "subject": subj_v,
            "body": body_v,
            "true_category": category,
            "true_priority": priority,
            "true_department": dept,
            "needs_draft": needs_draft,
            "has_money_terms": detect_has_money_terms(text),
            "has_security_terms": detect_has_security_terms(text),
            "has_link": detect_has_link(text),
            "has_urgency": detect_has_urgency(text),
            "multi_intent": detect_multi_intent(text),
            "persona": persona,
            "language_variant": lang,
            "intent_primary": intent_primary,
            "intent_secondary": intent_secondary,
            "difficulty_score": round(difficulty_score, 3),
            "difficulty": difficulty,
            # reward-aware expected action (same as true by default)
            "expected_action": {
                "category": category,
                "priority": priority,
                "department": dept
            }
        }
        examples.append(example)
    return examples[:n_per_category]

# deterministic confusing cases
def generate_confusing_cases(n: int = DEFAULT_CONFUSING) -> List[Dict]:
    cases = []
    templates = [
        ("Charged and app broken", "I was charged twice and now the app is not working after the update."),
        ("Refund requested but error", "I asked for a refund but now I can't log in; what's happening?"),
        ("Invoice shows error", "Invoice failed to load and my payment didn't go through."),
        ("Login link charged me", "I clicked a link and now I see a charge on my card and can't access account."),
        ("Service down and billed", "Service is down and I was billed; I want a refund and fix."),
        ("Payment failed and crash", "Payment failed during checkout and the site crashed; charged but no order."),
        ("Refund + bug", "I requested a refund and now the app shows an error when I try to check status."),
        ("Double charge + login issue", "I was charged twice and now I can't log in to see my invoices.")
    ]
    for i in range(n):
        subj, body = random.choice(templates)
        subj_v = noisy_variant(subj)
        body_v = noisy_variant(body)
        body_v = add_signature_variants(body_v)
        body_v = add_email_thread(body_v)
        body_v = add_hinglish(body_v)
        body_v = make_long_email(body_v)
        text = subj_v + " " + body_v
        t = text.lower()
        if (any(w in t for w in ["charge", "charged", "invoice", "payment"]) and any(w in t for w in ["not working", "error", "crash", "can't log"])):
            true_cat = "billing"; pr, dp = "P1", "billing_team"
        elif any(w in t for w in ["not working", "error", "crash", "can't log"]):
            true_cat = "technical"; pr, dp = "P2", "tech_support"
        elif any(w in t for w in ["refund", "return", "money back"]):
            true_cat = "refund"; pr, dp = "P2", "returns"
        elif any(w in t for w in SECURITY_TERMS):
            true_cat = "phishing"; pr, dp = "P1", "tech_support"
        else:
            if detect_has_money_terms(text):
                true_cat = "billing"; pr, dp = "P1", "billing_team"
            elif detect_has_security_terms(text):
                true_cat = "phishing"; pr, dp = "P1", "tech_support"
            else:
                true_cat = "technical"; pr, dp = "P2", "tech_support"
        needs_draft = true_cat in ("phishing", "refund", "complaint", "technical")
        example = {
            "subject": subj_v,
            "body": body_v,
            "true_category": true_cat,
            "true_priority": pr,
            "true_department": dp,
            "needs_draft": needs_draft,
            "has_money_terms": detect_has_money_terms(text),
            "has_security_terms": detect_has_security_terms(text),
            "has_link": detect_has_link(text),
            "has_urgency": detect_has_urgency(text),
            "multi_intent": detect_multi_intent(text),
            "persona": random.choice(PERSONAS),
            "language_variant": random.choice(LANG_VARIANTS),
            "intent_primary": true_cat,
            "intent_secondary": None,
            "difficulty_score": 0.8,
            "difficulty": "hard",
            "expected_action": {
                "category": true_cat,
                "priority": pr,
                "department": dp
            }
        }
        cases.append(example)
    return cases

# adversarial cases that trick rules
def generate_adversarial_cases(n: int = DEFAULT_ADVERSARIAL) -> List[Dict]:
    cases = []
    templates = [
        ("Payment Issue - Action Required", "Your invoice failed. Click http://secure-update.com to fix immediately and avoid suspension."),
        ("Invoice Payment Failed", "Your payment failed. Click here to update card details immediately."),
        ("App error - verify account", "App error occurred. Login again here: http://fake-link.com to resolve."),
        ("Refund and complaint", "Worst service ever. I want my money back NOW. Click link to confirm."),
    ]
    for _ in range(n):
        subj, body = random.choice(templates)
        subj_v = noisy_variant(subj)
        body_v = noisy_variant(body)
        body_v = add_signature_variants(body_v)
        body_v = add_email_thread(body_v)
        body_v = add_hinglish(body_v)
        body_v = make_long_email(body_v)
        text = subj_v + " " + body_v
        true_cat = "phishing"  # adversarial override
        pr, dp = "P1", "tech_support"
        example = {
            "subject": subj_v,
            "body": body_v,
            "true_category": true_cat,
            "true_priority": pr,
            "true_department": dp,
            "needs_draft": True,
            "has_money_terms": detect_has_money_terms(text),
            "has_security_terms": detect_has_security_terms(text),
            "has_link": detect_has_link(text),
            "has_urgency": detect_has_urgency(text),
            "multi_intent": True,
            "persona": random.choice(PERSONAS),
            "language_variant": random.choice(LANG_VARIANTS),
            "intent_primary": true_cat,
            "intent_secondary": "billing",
            "difficulty_score": 0.9,
            "difficulty": "hard",
            "expected_action": {"category": true_cat, "priority": pr, "department": dp}
        }
        cases.append(example)
    return cases

# negative samples and edge cases
def generate_negative_samples(n: int = DEFAULT_NEGATIVE) -> List[Dict]:
    samples = []
    templates = [
        ("Hello", "Just checking if everything is working fine."),
        ("Quick question", "Hi team, just wanted to know if the feature is available."),
        ("FYI", "Sharing a quick update about our usage."),
        ("Meeting follow-up", "Thanks for the meeting. No action required."),
        ("HELP!!!", "!!! 😡😡😡"),
        ("Only link", "http://example.com"),
        ("Empty body", ""),
    ]
    for _ in range(n):
        subj, body = random.choice(templates)
        subj_v = noisy_variant(subj)
        body_v = noisy_variant(body)
        body_v = add_signature_variants(body_v)
        body_v = add_hinglish(body_v)
        text = subj_v + " " + body_v
        example = {
            "subject": subj_v,
            "body": body_v,
            "true_category": "general",
            "true_priority": "P3",
            "true_department": "customer_success",
            "needs_draft": False,
            "has_money_terms": detect_has_money_terms(text),
            "has_security_terms": detect_has_security_terms(text),
            "has_link": detect_has_link(text),
            "has_urgency": detect_has_urgency(text),
            "multi_intent": detect_multi_intent(text),
            "persona": random.choice(PERSONAS),
            "language_variant": random.choice(LANG_VARIANTS),
            "intent_primary": "general",
            "intent_secondary": None,
            "difficulty_score": 0.1,
            "difficulty": "easy",
            "expected_action": {"category": "general", "priority": "P3", "department": "customer_success"}
        }
        samples.append(example)
    return samples

# --- distribution shift simulation ---
def simulate_distribution_shift(dataset: List[Dict], shift_type: str = "phishing_spike") -> List[Dict]:
    if shift_type == "phishing_spike":
        phishing = [d for d in dataset if d["true_category"] == "phishing"]
        return dataset + random.choices(phishing, k=len(dataset)//2)
    elif shift_type == "billing_season":
        billing = [d for d in dataset if d["true_category"] == "billing"]
        return dataset + random.choices(billing, k=len(dataset)//3)
    elif shift_type == "adversarial_wave":
        adv = [d for d in dataset if d.get("difficulty") == "hard"]
        return dataset + random.choices(adv, k=len(dataset)//3)
    return dataset

# --- label noise injection ---
def inject_label_noise(dataset: List[Dict], noise_prob: float = LABEL_NOISE_PROB) -> List[Dict]:
    for d in dataset:
        if random.random() < noise_prob:
            d["true_category"] = random.choice(CATEGORIES)
            # update expected_action to match noisy label
            if d.get("true_category"):
                cat = d["true_category"]
                if cat == "phishing":
                    pr, dp = "P1", "tech_support"
                elif cat == "billing":
                    pr, dp = "P1", "billing_team"
                elif cat == "technical":
                    pr, dp = "P2", "tech_support"
                elif cat == "refund":
                    pr, dp = "P2", "returns"
                elif cat == "complaint":
                    pr, dp = "P3", "customer_success"
                else:
                    pr, dp = "P3", "customer_success"
                d["expected_action"] = {"category": cat, "priority": pr, "department": dp}
    return dataset

# --- persistence and helpers ---
def build_and_save_dataset(n_per_category: int = DEFAULT_PER_CATEGORY,
                           confusing_cases: int = DEFAULT_CONFUSING,
                           adversarial_cases: int = DEFAULT_ADVERSARIAL,
                           negative_samples: int = DEFAULT_NEGATIVE,
                           out_json: str = OUT_JSON) -> List[Dict]:
    dataset = []
    for cat in CATEGORIES:
        dataset.extend(generate_examples_per_category(cat, n_per_category))
    dataset.extend(generate_confusing_cases(n=confusing_cases))
    dataset.extend(generate_adversarial_cases(n=adversarial_cases))
    dataset.extend(generate_negative_samples(n=negative_samples))
    random.shuffle(dataset)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    return dataset

def load_dataset(path: str = OUT_JSON) -> List[Dict]:
    if not os.path.exists(path):
        return build_and_save_dataset()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_episode_sample(phase: int = 1, dataset: Optional[List[Dict]] = None, k: int = 1) -> List[Dict]:
    if dataset is None:
        dataset = load_dataset()
    if phase == 1:
        candidates = [d for d in dataset if d.get("difficulty") == "easy"]
    elif phase == 2:
        candidates = [d for d in dataset if d.get("difficulty") in ("easy", "medium")]
    else:
        candidates = dataset
    random.shuffle(candidates)
    return candidates[:k]

class DatasetEnvHelper:
    def __init__(self, dataset: List[Dict]):
        self.dataset = dataset[:]
        self.index = 0
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.dataset)
        self.index = 0

    def pop_next(self) -> Dict:
        if self.index >= len(self.dataset):
            self.shuffle()
        item = self.dataset[self.index]
        self.index += 1
        return copy.deepcopy(item)

# --- edge case library export ---
def build_edge_case_library() -> List[Dict]:
    return generate_negative_samples(n=50)

# --- if run as script ---
if __name__ == "__main__":
    ds = build_and_save_dataset()
    counts = {}
    for d in ds:
        counts[d["true_category"]] = counts.get(d["true_category"], 0) + 1
    print("Dataset built:", OUT_JSON)
    print("Counts per category:", counts)
    print("Total examples:", len(ds))
