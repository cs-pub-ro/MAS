#!/bin/bash
# Simple runner that connects to MCP servers

set -e

echo "=============================================="
echo "MCP House Building Simulation"
echo "=============================================="
echo ""
echo "Make sure you have started the servers in other terminals:"
echo ""
echo "  Terminal 1: python -m auction_server.mcp_server"
echo "  Terminal 2: python -m negotiation_server.mcp_server"
echo ""
echo "=============================================="
echo ""

python orchestrator.py
