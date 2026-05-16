"""Client adapter for Auction MCP server.

When running with real MCP servers, use:
    client = AuctionServerClient(stdio_process)

When running in-process (for testing/development), use:
    client = AuctionServerClient(mcp_server_instance)
"""

import json
from typing import Any, Dict, List


class AuctionServerClient:
    """Adapter for calling AuctionMCPServer (via MCP or in-process)."""

    def __init__(self, server):
        """Initialize with either an MCP server instance or stdio_process."""
        self.server = server

    def get_available_tools(self, agent_name: str) -> List[Dict[str, Any]]:
        """Return tools available to this agent."""
        # In-process version
        if hasattr(self.server, "_list_tools"):
            tools = self.server._list_tools()
            return [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema,
                }
                for t in tools
            ]
        # MCP version would use client.call_tool()
        raise NotImplementedError("MCP client not yet implemented")

    def start_auction(self, task: Any) -> str:
        """Initialize auction for a task."""
        if hasattr(self.server, "start_auction"):
            return self.server.start_auction(task)
        raise NotImplementedError("MCP client not yet implemented")

    def execute_tool(
        self, agent_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool for an agent."""
        if hasattr(self.server, "_call_tool"):
            # In-process: call async method synchronously
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                self.server._call_tool(tool_name, arguments)
            )
            # result is list[TextContent], extract JSON
            if result and len(result) > 0:
                return json.loads(result[0].text)
            return {"status": "error", "message": "No result"}
        raise NotImplementedError("MCP client not yet implemented")

    def get_state(self) -> Dict[str, Any]:
        """Get current auction state."""
        if hasattr(self.server, "get_state"):
            return self.server.get_state()
        raise NotImplementedError("MCP client not yet implemented")
