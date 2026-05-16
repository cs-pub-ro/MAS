"""MCP server for auction operations using SSE (Server-Sent Events) HTTP transport.

Real MCP implementation using aiohttp with SSE for bidirectional communication.
Run with: python -m auction_server.server
"""

from mcp.server.fastmcp import FastMCP
from auction_server.state import AuctionState
from auction_server.tools import AuctionTools
from shared.types import Task, AgentState
from shared.logger import StructuredLogger
import uuid
from typing import Any, Dict


server = FastMCP("auction-server")
_auctions_store: Dict[str, AuctionState] = {}
logger = StructuredLogger("AuctionServer")


@server.tool()
def start_auction(task_name: str, budget: float) -> Any:
    """Start a new auction for a task."""
    session_id = str(uuid.uuid4())[:8]
    auction = AuctionState(
        session_id=session_id,
        task=Task(name=task_name, budget=budget)
    )
    _auctions_store[task_name] = auction

    for company in ["A", "B", "C", "D", "E", "F"]:
        auction.agent_states[company] = AgentState(
            name=company, joined=False, active=False
        )

    logger.log_info(f"Auction started for task '{task_name}' (budget: {budget})")
    return {"status": "success", "session_id": session_id}


@server.tool()
def propose_budget(task_name: str, price: float) -> Any:
    """ACME proposes a budget price for auction round."""
    if task_name not in _auctions_store:
        return {"status": "error", "message": f"No auction for task '{task_name}'"}

    auction = _auctions_store[task_name]
    tools = AuctionTools(auction)
    return tools.propose_budget(price)


@server.tool()
def bid(task_name: str, company: str) -> Any:
    """Company submits a bid at current proposed price."""
    if task_name not in _auctions_store:
        return {"status": "error", "message": f"No auction for task '{task_name}'"}

    auction = _auctions_store[task_name]
    tools = AuctionTools(auction)
    return tools.bid(company)


@server.tool()
def get_status(task_name: str) -> Any:
    """Get current auction status."""
    if task_name not in _auctions_store:
        return {"status": "error", "message": f"No auction for task '{task_name}'"}

    auction = _auctions_store[task_name]
    tools = AuctionTools(auction)
    return tools.get_status()


if __name__ == "__main__":
    server.settings.host = '127.0.0.1'
    server.settings.port = 8010
    server.run(transport='sse')
