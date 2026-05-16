"""
LLM client + prompt loader + cost tracking for the MAS House-Building lab.

All OpenAI plumbing lives in this single file. Students normally edit only the
text files in ``prompts/`` to change agent behaviour.

Public surface:
    - call_llm(...)        : run a prompt, parse JSON, return value, log cost
    - render_prompt(...)   : load prompts/<name>.txt and substitute $placeholders
    - session_summary()    : cumulative tokens / cost for the current run

Behaviour:
    - If OPENAI_API_KEY is missing the very first call_llm() raises and the lab
      stops. No fallback heuristics — students see real LLM behaviour or no
      simulation at all.
    - Network / parse errors propagate too. Re-run the lab.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from string import Template
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment + globals
# ---------------------------------------------------------------------------

load_dotenv()  # populate os.environ from .env if present

logger = logging.getLogger("llm")

PROMPTS_DIR = Path(__file__).parent / "prompts"

DEFAULT_MODEL       = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "500"))

# Prices in USD per 1M tokens (input, output). Update if OpenAI changes pricing.
MODEL_PRICES = {
    "gpt-4o-mini":   (0.15,  0.60),
    "gpt-4o":        (2.50, 10.00),
    "gpt-4.1":       (2.00,  8.00),
    "gpt-4.1-mini":  (0.40,  1.60),
    "gpt-4.1-nano":  (0.10,  0.40),
    "o4-mini":       (1.10,  4.40),
}

_session_cost_usd: float = 0.0
_session_calls: int = 0
_session_tokens: int = 0


def session_summary() -> Dict[str, Any]:
    return {
        "calls": _session_calls,
        "tokens": _session_tokens,
        "cost_usd": round(_session_cost_usd, 6),
    }


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    if model in MODEL_PRICES:
        inp, out = MODEL_PRICES[model]
    else:
        match = next((v for k, v in MODEL_PRICES.items() if model.startswith(k)), None)
        if match is None:
            return 0.0
        inp, out = match
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _stringify(v: Any) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, default=str, indent=2, ensure_ascii=False)
    return str(v)


def render_prompt(name: str, context: Dict[str, Any]) -> str:
    """Load prompts/<name>.txt and substitute $placeholders with context values.

    Uses ``string.Template.safe_substitute`` so unknown $vars are left intact
    rather than raising — students can introduce new placeholders gradually.
    """
    tpl = Template(load_prompt(name))
    return tpl.safe_substitute({k: _stringify(v) for k, v in context.items()})


# ---------------------------------------------------------------------------
# OpenAI client (lazy)
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        _client = OpenAI(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are the strategic decision module of an autonomous agent in a "
    "multi-agent house-building negotiation game. You always reply with a "
    "single valid JSON object, with no markdown fencing and no extra prose, "
    "matching the schema requested in the user prompt."
)


def _coerce_value(value: Any, expect: str) -> Any:
    if expect == "float":
        if value is None:
            raise ValueError("LLM returned null where a number was expected")
        return float(value)
    if expect == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in ("true", "yes", "1", "y", "accept")
        raise ValueError(f"Cannot coerce {value!r} to bool")
    return value


# Human-readable explanation of what each agent function's return value means.
# Used by the [LLM] log line so the reader knows what the chosen value represents.
DECISION_LABELS = {
    "propose_item_budget":       "price ACME announces to companies in this auction round",
    "provide_negotiation_offer": "ACME's offer (initiator side) in this negotiation round",
    "decide_bid":                "accept the announced auction price (True=accept, False=wait)",
    "respond_to_offer":          "company's counter-offer (asking price) in this negotiation round",
}


def _format_reasoning(text: Any) -> str:
    """Collapse internal whitespace so reasoning fits on a single log line.

    We do NOT truncate — the user wants to see the full LLM thinking.
    """
    if text is None:
        return "(none)"
    return " ".join(str(text).split())


def call_llm(
    prompt_name: str,
    agent_name: str,
    function_name: str,
    context: Dict[str, Any],
    expect: str,                # "float" | "bool"
    model: str | None = None,
) -> Any:
    """Render a prompt, call OpenAI, parse JSON, log a one-liner, return value.

    Errors propagate (no retry, no fallback). The lab will crash loudly if
    the key is missing, the network fails, or the LLM returns bad JSON.
    """
    global _session_cost_usd, _session_calls, _session_tokens

    model = model or DEFAULT_MODEL
    user_prompt = render_prompt(prompt_name, context)

    # Verbose by intent: only emitted when the `llm` logger is set to DEBUG.
    logger.debug("[LLM PROMPT] %s.%s\n%s", agent_name, function_name, user_prompt)

    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )

    content = resp.choices[0].message.content or "{}"
    usage   = resp.usage
    pt, ct, tt = usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
    cost    = compute_cost(model, pt, ct)

    _session_calls   += 1
    _session_tokens  += tt
    _session_cost_usd += cost

    data  = json.loads(content)
    value = _coerce_value(data.get("value"), expect)

    label    = DECISION_LABELS.get(function_name, "decision")
    thinking = _format_reasoning(data.get("reasoning"))

    logger.info(
        "[LLM call] %s.%s\n"
        "    thinking : %s\n"
        "    chose    : %s   (meaning: %s)\n"
        "    usage    : %d tokens (in=%d, out=%d), cost $%.5f\n"
        "    session  : %d calls so far, total cost $%.5f",
        agent_name, function_name,
        thinking,
        value, label,
        tt, pt, ct, cost,
        _session_calls, _session_cost_usd,
    )
    return value
