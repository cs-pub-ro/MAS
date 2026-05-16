"""LLM configuration for agents."""

import os
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class LLMConfig:
    """LLM configuration."""

    model: str
    reasoning_effort: Literal["low", "medium", "high"]
    is_reasoning_model: bool
    max_completion_tokens: Optional[int] = None  # Only for non-reasoning models
    temperature: Optional[float] = None  # Only for non-reasoning models

    def to_dict(self):
        """Convert to API request params."""
        params = {
            "model": self.model,
        }

        if self.is_reasoning_model:
            params["reasoning_effort"] = self.reasoning_effort
        else:
            params["temperature"] = self.temperature
            if self.max_completion_tokens:
                params["max_completion_tokens"] = self.max_completion_tokens

        return params


# Default configuration: GPT-5-nano with medium reasoning
DEFAULT_LLM_CONFIG = LLMConfig(
    model="gpt-5-nano",
    reasoning_effort="medium",
    is_reasoning_model=True,
)

# Alternative configurations
FAST_CONFIG = LLMConfig(
    model="gpt-4-turbo",
    reasoning_effort="low",
    is_reasoning_model=False,
    max_completion_tokens=8000,
    temperature=0.7,
)

DEEP_REASONING_CONFIG = LLMConfig(
    model="gpt-5-nano",
    reasoning_effort="high",
    is_reasoning_model=True,
)


def get_llm_config(config_name: str = "default") -> LLMConfig:
    """Get LLM configuration by name."""
    configs = {
        "default": DEFAULT_LLM_CONFIG,
        "fast": FAST_CONFIG,
        "deep": DEEP_REASONING_CONFIG,
    }

    selected = configs.get(config_name, DEFAULT_LLM_CONFIG)

    # Allow override via environment variable
    if os.getenv("LLM_CONFIG"):
        return configs.get(os.getenv("LLM_CONFIG"), selected)

    return selected
