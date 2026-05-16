#!/bin/bash
# Start the MCP servers in background

echo "Starting Auction Server..."
python -m auction_server.mcp_server &
AUCTION_PID=$!

echo "Starting Negotiation Server..."
python -m negotiation_server.mcp_server &
NEGOTIATION_PID=$!

echo "Servers started:"
echo "  Auction Server (PID: $AUCTION_PID)"
echo "  Negotiation Server (PID: $NEGOTIATION_PID)"
echo ""
echo "To stop servers, run: kill $AUCTION_PID $NEGOTIATION_PID"
