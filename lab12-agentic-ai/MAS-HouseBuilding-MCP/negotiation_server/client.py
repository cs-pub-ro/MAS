"""Client adapter for Negotiation MCP server.

When running with real MCP servers, use:
    client = NegotiationServerClient(stdio_process)

When running in-process (for testing/development), use:
    client = NegotiationServerClient(mcp_server_instance)
"""

import json
from typing import Any, Dict, List


class NegotiationServerClient:
    """Adapter for calling NegotiationMCPServer (via MCP or in-process)."""

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

    def start_negotiation(self, task: Any, bidders: List[str]) -> str:
        """Initialize negotiation session for a task."""
        if hasattr(self.server, "start_negotiation"):
            return self.server.start_negotiation(task, bidders)
        raise NotImplementedError("MCP client not yet implemented")

    def set_current_task(self, task_name: str):
        """Set the current task being negotiated."""
        if hasattr(self.server, "set_current_task"):
            return self.server.set_current_task(task_name)
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

    def get_state(self, task_name: str = None) -> Dict[str, Any]:
        """Get state for a specific negotiation."""
        if hasattr(self.server, "get_state"):
            return self.server.get_state(task_name)
        raise NotImplementedError("MCP client not yet implemented")
