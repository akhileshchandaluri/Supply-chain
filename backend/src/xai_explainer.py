"""
xai_explainer.py — SmartChain AI "Layer 6" Explainable-AI Module

Translates the raw pipeline JSON (demand forecast, risk, anomaly, RL action,
optimization allocation, routing) into a concise, human-readable executive
summary using the Groq chat-completions API (Llama 3.3 70B).

The LLM is constrained to speak like a logistics auditor — plain operational
language, no ML jargon — so the output can be surfaced directly on the dashboard.
"""

import os
import requests

# Load backend/.env so GROQ_API_KEY is available regardless of entry point
# (uvicorn server, integration import, or a standalone test script). A plain
# .env file does nothing until something loads it into os.environ.
try:
    from dotenv import load_dotenv

    _ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(_ENV_PATH)
except ImportError:
    # python-dotenv not installed — fall back to whatever is already in the
    # process environment (e.g. a real exported env var in production).
    pass

# ─── Groq API configuration ───────────────────────────────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT = 5  # seconds — keep short so the pipeline never blocks on a slow network
FALLBACK_MESSAGE = "Audit generation unavailable"

SYSTEM_PROMPT = (
    "You are a Lead AI Logistics Auditor. You review the automated decisions of "
    "a supply-chain control system and explain them to executives.\n"
    "Rules you MUST follow:\n"
    "1. Respond in EXACTLY 2 to 3 sentences — no more, no less.\n"
    "2. Use NO technical machine-learning jargon. Never mention Q-tables, epochs, "
    "weights, models, algorithms, or training.\n"
    "3. Focus solely on operational logic, risk mitigation, and cost efficiency.\n"
    "Write in a confident, professional, board-room tone."
)


def _format_pipeline_prompt(pipeline_data: dict) -> str:
    """Flatten the pipeline dict into a clear natural-language brief for the LLM."""
    demand = pipeline_data.get("demand_7d_avg", "N/A")
    risk_level = pipeline_data.get("risk_level", "N/A")

    anomaly = pipeline_data.get("anomaly", {})
    anomaly_state = "detected" if anomaly.get("is_anomaly") else "none detected"

    rl_action = pipeline_data.get("rl_action", {})
    action = rl_action.get("action", "N/A") if isinstance(rl_action, dict) else rl_action

    route = pipeline_data.get("route", {}) or {}
    route_type = route.get("type", "N/A")
    is_emergency = pipeline_data.get("is_emergency", False)

    return (
        "Here is the current supply-chain decision snapshot to audit:\n"
        f"- Projected 7-day demand (avg/day): {demand} units\n"
        f"- Delivery risk level: {risk_level}\n"
        f"- Supplier irregularity: {anomaly_state}\n"
        f"- Recommended inventory action: {action}\n"
        f"- Fulfilment mode: {route_type} "
        f"({'emergency' if is_emergency else 'standard'})\n\n"
        "Write the executive audit summary now."
    )


def _call_groq(messages: list, temperature: float = 0.3, max_tokens: int = 200,
               timeout: int = REQUEST_TIMEOUT) -> str:
    """
    Shared Groq chat-completions call. Returns the assistant text, or
    FALLBACK_MESSAGE on missing key / timeout / network / malformed response.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return FALLBACK_MESSAGE

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except (requests.exceptions.Timeout, requests.exceptions.RequestException):
        return FALLBACK_MESSAGE
    except (KeyError, IndexError, ValueError):
        # Malformed / unexpected response body.
        return FALLBACK_MESSAGE


def generate_pipeline_explanation(pipeline_data: dict) -> str:
    """
    Produce a 2-3 sentence executive summary of the pipeline decision via Groq.
    Returns FALLBACK_MESSAGE on any failure so callers never handle exceptions.
    """
    return _call_groq(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _format_pipeline_prompt(pipeline_data)},
        ],
        temperature=0.3,
        max_tokens=200,
    )


# ─── Interactive RL decision chat ─────────────────────────────────────────────
CHAT_SYSTEM_PROMPT = (
    "You are an RL Systems Analyst embedded in a supply-chain control tower. You "
    "explain the decisions of a tabular Q-Learning inventory agent to an engineer, "
    "grounding every answer in the data you are given (Q-values, the itemized reward "
    "breakdown, the chosen action, and the optimization/risk context).\n\n"
    "The agent chooses the action with the highest Q-value (expected long-term "
    "discounted reward). The one-step reward function is:\n"
    "  reward = -0.05*inventory_after            (holding cost)\n"
    "           +0.5*demand   if fulfilled       (service reward)\n"
    "           -50           if stockout         (shortage penalty)\n"
    "           -20           if inventory>1000   (overcapacity)\n"
    "           -15 emergency / -2 standard reorder (action cost)\n"
    "           +10           for switching supplier during an anomaly.\n\n"
    "Rules:\n"
    "1. Be concise (2-5 sentences) and concrete — cite the actual numbers.\n"
    "2. For hypothetical/'what-if' questions, reason from the reward rules above; "
    "be explicit that you are reasoning about how the score WOULD change, since you "
    "only have the current state's real Q-values.\n"
    "3. No fabricated numbers. If the data doesn't support a claim, say so.\n"
    "4. Plain operational language — minimal jargon."
)


def chat_about_decision(question: str, context: dict, history=None) -> str:
    """
    Answer a user question about the latest RL/pipeline decision via Groq,
    grounded in the provided run context. Returns FALLBACK_MESSAGE on failure.
    """
    import json

    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    # Ground the conversation with the real run data.
    messages.append({
        "role": "user",
        "content": "Here is the latest pipeline decision data (JSON):\n"
                   + json.dumps(context, indent=2),
    })
    messages.append({
        "role": "assistant",
        "content": "Understood — I have the decision data and will answer using it.",
    })
    # Replay recent turns for follow-up coherence.
    for turn in (history or [])[-6:]:
        role = turn.get("role")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})

    return _call_groq(messages, temperature=0.4, max_tokens=400, timeout=15)

