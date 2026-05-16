#!/bin/bash
set -e

echo "Starting servers in background..."

# Start servers in background
conda run -n ami-agents python -m auction_server.mcp_server > /tmp/auction.log 2>&1 &
AUCTION_PID=$!
echo "Auction server started (PID: $AUCTION_PID)"

conda run -n ami-agents python -m negotiation_server.mcp_server > /tmp/negotiation.log 2>&1 &
NEGOTIATION_PID=$!
echo "Negotiation server started (PID: $NEGOTIATION_PID)"

# Give servers time to start
sleep 3

# Run orchestrator
echo "Starting orchestrator..."
echo "" | conda run -n ami-agents python orchestrator.py 2>&1 | head -50

# Kill servers
kill $AUCTION_PID $NEGOTIATION_PID 2>/dev/null || true
echo "Done"
