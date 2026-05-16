import json
import os
from typing import Dict, Any
from shared.types import AgentState
from config.llm_config import get_llm_config


class BaseAgent:
    def __init__(self, name: str, client):
        self.name = name
        self.client = client
        self.memory: Dict[str, Any] = {
            "last_action": "",
            "last_offer_seen": "",
        }
        self.auction_server = None
        self.negotiation_server = None
        self.llm_config = get_llm_config(os.getenv("LLM_CONFIG", "default"))

    def register(self, auction_server, negotiation_server):
        """Register with MCP servers."""
        self.auction_server = auction_server
        self.negotiation_server = negotiation_server

    def think(self, prompt: str) -> str:
        """Call OpenAI LLM to get decision."""
        request_params = self.llm_config.to_dict()
        request_params["messages"] = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        response = self.client.chat.completions.create(**request_params)

        # Extract text from response
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Handle message with thinking blocks
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block.get("text", "")
                    elif hasattr(block, "type") and block.type == "text":
                        return block.text

        return ""

    def decide(self, prompt: str) -> Dict[str, Any]:
        """Get LLM decision and parse JSON."""
        response = self.think(prompt)

        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            decision = json.loads(text)
            return decision
        except (json.JSONDecodeError, IndexError) as e:
            return {
                "tool": "get_status",
                "arguments": {},
                "error": f"Failed to parse response: {e}",
            }

    def get_memory_state(self) -> Dict[str, Any]:
        """Return minimal memory state."""
        return {
            "last_action": self.memory.get("last_action", ""),
            "last_offer_seen": self.memory.get("last_offer_seen", ""),
        }

    def update_memory(self, action: str, offer_price: str = ""):
        """Update local memory after action."""
        self.memory["last_action"] = action
        if offer_price:
            self.memory["last_offer_seen"] = offer_price
