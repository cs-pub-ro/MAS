"""Negotiation MCP server with SSE HTTP transport.

Run with: python -m negotiation_server.server
Accessible at: http://127.0.0.1:8011/sse
"""

from typing import Dict, Any
from mcp.server.fastmcp import FastMCP
from negotiation_server.state import NegotiationState
from negotiation_server.tools import NegotiationTools
from shared.types import Task, AgentState
from shared.logger import StructuredLogger
import uuid

server = FastMCP("negotiation-server")
negotiations: Dict[str, NegotiationState] = {}
current_task: str = None
logger = StructuredLogger("NegotiationServer")


@server.tool()
def start_negotiation(task_name: str, budget: float, bidders: list) -> Any:
    """Start a new negotiation for a task with selected bidders."""
    global current_task, negotiations

    session_id = str(uuid.uuid4())[:8]
    negotiation = NegotiationState(
        session_id=session_id,
        task=Task(name=task_name, budget=budget),
        bidders=bidders,
    )
    negotiations[task_name] = negotiation
    current_task = task_name

    for agent_name in ["ACME"] + bidders:
        negotiation.agent_states[agent_name] = AgentState(
            name=agent_name, joined=True, active=True
        )

    logger.log_info(f"Negotiation started for task '{task_name}' with bidders {bidders}")
    return {"status": "success", "session_id": session_id}


@server.tool()
def make_offer(price: float, type: str, from_: str = "ACME", to_: str = "") -> Any:
    """Make an offer in negotiation (offer, counter, or accept)."""
    global current_task, negotiations

    if not current_task or current_task not in negotiations:
        return {"status": "error", "message": "No active negotiation"}

    negotiation = negotiations[current_task]
    tools = NegotiationTools(negotiation)

    return tools.make_offer(
        from_agent=from_,
        to_agent=to_,
        price=price,
        offer_type=type,
    )


@server.tool()
def get_status() -> Any:
    """Get current negotiation status."""
    global current_task, negotiations

    if not current_task or current_task not in negotiations:
        return {"status": "error", "message": "No active negotiation"}

    return negotiations[current_task].to_dict()


if __name__ == "__main__":
    server.settings.host = '127.0.0.1'
    server.settings.port = 8011
    server.run(transport='sse')
