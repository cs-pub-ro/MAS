#!/usr/bin/env python3
"""Verify that all dependencies are installed."""

import sys
import subprocess

required_packages = {
    "mcp": "mcp>=1.0.0",
    "openai": "openai>=1.0.0",
    "pyyaml": "pyyaml>=6.0",
    "dotenv": "python-dotenv>=1.0.0",
}

missing = []
for module, requirement in required_packages.items():
    try:
        __import__(module)
        print(f"✓ {module} is installed")
    except ImportError:
        print(f"✗ {module} is NOT installed")
        missing.append(requirement)

if missing:
    print("\nMissing packages. Install with:")
    print(f"pip install {' '.join(missing)}")
    sys.exit(1)
else:
    print("\n✓ All dependencies are installed!")
    print("\nYou can now run:")
    print("  Terminal 1: python -m auction_server.mcp_server")
    print("  Terminal 2: python -m negotiation_server.mcp_server")
    print("  Terminal 3: python orchestrator.py")
    sys.exit(0)
