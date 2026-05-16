"""Auction MCP server with SSE HTTP transport.

Run with: python -m auction_server.server
Accessible at: http://127.0.0.1:8010/sse
"""

from mcp.server.fastmcp import FastMCP
from auction_server.state import AuctionState
from auction_server.tools import AuctionTools
from shared.types import Task, AgentState
from shared.logger import StructuredLogger
import uuid
from typing import Any

server = FastMCP("auction-server")
current_auction: AuctionState = None
tools: AuctionTools = None
logger = StructuredLogger("AuctionServer")


@server.tool()
def start_auction(task_name: str, budget: float) -> Any:
    """Start a new auction for a task."""
    global current_auction, tools

    session_id = str(uuid.uuid4())[:8]
    current_auction = AuctionState(
        session_id=session_id,
        task=Task(name=task_name, budget=budget)
    )
    tools = AuctionTools(current_auction)

    for company in ["A", "B", "C", "D", "E", "F"]:
        current_auction.agent_states[company] = AgentState(
            name=company, joined=False, active=False
        )

    logger.log_info(f"Auction started for task '{task_name}' (budget: {budget})")
    return {"status": "success", "session_id": session_id}


@server.tool()
def propose_budget(price: float) -> Any:
    """ACME proposes a budget price for auction round."""
    global current_auction, tools
    if not current_auction or not tools:
        return {"status": "error", "message": "No active auction"}
    return tools.propose_budget(price)


@server.tool()
def bid(company: str) -> Any:
    """Company submits a bid at current proposed price."""
    global current_auction, tools
    if not current_auction or not tools:
        return {"status": "error", "message": "No active auction"}
    return tools.bid(company)


@server.tool()
def get_status() -> Any:
    """Get current auction status."""
    global current_auction, tools
    if not current_auction or not tools:
        return {"status": "error", "message": "No active auction"}
    return tools.get_status()


if __name__ == "__main__":
    server.settings.host = '127.0.0.1'
    server.settings.port = 8010
    server.run(transport='sse')
