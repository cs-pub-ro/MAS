#!/usr/bin/env python3
"""Test that MCP servers can be started and connected to."""

import asyncio
import subprocess
import time
from mcp.client.stdio import stdio_client, StdioServerParameters

async def test():
    print("Starting auction server in background...")
    proc = subprocess.Popen(
        ["python", "-m", "auction_server.mcp_server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(1)  # Give server time to start
    
    try:
        print("Connecting to auction server...")
        async with stdio_client(StdioServerParameters(
            command="python",
            args=["-m", "auction_server.mcp_server"]
        )) as client:
            print("✓ Connected to auction server")
            
            tools = await client.list_tools()
            print(f"✓ Listed {len(tools)} tools: {[t.name for t in tools]}")
            
            result = await client.call_tool("start_auction", {
                "task_name": "test_task",
                "budget": 1000
            })
            print(f"✓ Called start_auction: {result}")
            
    finally:
        proc.terminate()
        proc.wait()
        print("✓ Test passed!")

if __name__ == "__main__":
    asyncio.run(test())
