# MAS House Building with MCP Multi-Agent Construction Contracting

A complete multi-agent simulation system modeling ACME outsourcing construction work to companies A–F using **real MCP (Model Context Protocol) servers**. Agents discover tools from servers and make LLM-driven decisions.

## How to Run

```bash
# Install dependencies (one time)
pip install -r requirements.txt

# Setup environment (one time)
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
nano .env

# Run the simulation
python orchestrator.py
```

The orchestrator automatically launches both FastMCP servers as subprocesses. The servers expose HTTP APIs on `localhost:8010` (auction) and `localhost:8011` (negotiation). The orchestrator connects to these HTTP endpoints and runs the complete auction and negotiation simulation with detailed logging.

### MCP Servers

#### Auction Server (`auction_server/mcp_server.py`)
- **Transport**: stdio
- **Tools**:
  - `propose_budget(price)` — ACME only
  - `bid(company)` — Companies only
  - `get_status()` — Everyone
- **Protocol**: Fixed 3 rounds, companies bid YES/NO at ACME's proposed prices

#### Negotiation Server (`negotiation_server/mcp_server.py`)
- **Transport**: stdio
- **Tools**:
  - `make_offer(price, type)` — Both ACME and companies
  - `get_status()` — Everyone
- **Protocol**: Up to 3 rounds, monotonic concession

### Agents (LLM-Driven)

- **ACME Agent**: Reads tool descriptions from server, decides what price to propose and what offers to make
- **Company Agents**: 6 independent instances (A–F), each reads tools and decides whether to bid or what offer to make

**Key**: Agents do NOT hardcode tool names. Prompts show available tools, and LLM decides which tool to call.

## Project Structure

```
MAS-HouseBuilding-MCP/
├── auction_server/
│   ├── mcp_server.py      # Real MCP server with async handlers
│   ├── client.py          # Client adapter for in-process or stdio
│   ├── state.py           # AuctionState dataclass
│   └── tools.py           # AuctionTools (propose_budget, bid, get_status)
├── negotiation_server/
│   ├── mcp_server.py      # Real MCP server with async handlers
│   ├── client.py          # Client adapter
│   ├── state.py           # NegotiationState dataclass
│   └── tools.py           # NegotiationTools (make_offer, get_status)
├── agents/
│   ├── base_agent.py      # Base agent with OpenAI client
│   ├── acme.py            # ACME buyer agent workflows
│   └── company.py         # Company contractor agent workflows
├── prompts/
│   ├── acme_auction_round.txt          # ACME auction decisions
│   ├── company_auction_round.txt       # Company auction decisions
│   ├── acme_negotiation_round.txt      # ACME negotiation decisions
│   └── company_negotiation_round.txt   # Company negotiation decisions
├── config/
│   ├── llm_config.py      # LLM configuration (gpt-5-nano default)
│   ├── acme_config.py     # ACME tasks and budget
│   └── contractors.yaml   # Company specialties and costs
├── shared/
│   ├── types.py           # Dataclasses (Task, Bid, Offer, etc.)
│   └── logger.py          # Structured logging
├── orchestrator.py        # Main entry point
├── requirements.txt       # Python dependencies
├── .env.example           # Environment template
├── ARCHITECTURE.md        # System architecture
└── README.md              # This file
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env
```

## Run Simulation

Simply run the orchestrator from the project root:

```bash
python orchestrator.py
```

The orchestrator will:
1. Launch the auction server as a subprocess
2. Launch the negotiation server as a subprocess
3. Wait for both servers to initialize and be ready
4. Execute the complete auction and negotiation simulation
5. Automatically clean up the server processes when done

## Configuration

Control LLM reasoning effort via environment variable:

```bash
# Medium reasoning (default)
python orchestrator.py

# Fast (no reasoning)
LLM_CONFIG=fast python orchestrator.py

# Deep reasoning
LLM_CONFIG=deep python orchestrator.py
```

## How It Works

### Auction Phase (Per Task)

1. ACME agent calls `run_auction_workflow([task])`
2. Server initializes auction state
3. For 3 rounds:
   - ACME's LLM reads available tools → decides price to propose
   - Server executes `propose_budget(price)`
   - Each company's LLM reads available tools → decides to bid or skip
   - Companies that bid YES are added to bidders list
4. All companies that bid YES proceed to negotiation

### Negotiation Phase (Per Task-Bidders Pair)

1. ACME agent calls `run_negotiation_workflow(auction_bidders, [task])`
2. Server initializes negotiation state with all bidders
3. For up to 3 rounds:
   - ACME's LLM reads available tools → decides offer (price + type)
   - Server executes `make_offer(price, type)`
   - Each company's LLM reads available tools → decides offer
   - Server executes company's `make_offer(price, type)`
   - If any offer is `accept`, negotiation ends
4. Final price and contractor recorded

## Key Design Points

1. **Tool Discovery**: Agents receive tool descriptions in prompts, decide which tool to call
2. **Response Format**: All agent responses are JSON: `{"tool": "...", "arguments": {...}}`
3. **Server as Source of Truth**: All state lives on server, agents only read/execute
4. **Minimal Agent Memory**: Only `last_action` and `last_offer_seen`, rest from server
5. **Protocol Enforcement**: Server enforces auction/negotiation rules, agents can't bypass
6. **Full Logging**: Every action logged with before/after state for replay

## Testing

Run the simulation a few times with different `LLM_CONFIG` values. You should see:

- All 4 tasks run through auction and negotiation
- Multiple companies bidding on same task (both auction and negotiation)
- Final contracts within budget
- Total savings displayed

Example output:
```
=== AUCTION PHASE: structural design ===
--- Round 1 ---
ACME proposes: $3000.00
Company A: Bid submitted
Company C: Bid submitted
Auction complete. Bidders: ['A', 'C']

=== NEGOTIATION PHASE: structural design ===
Negotiating with: A, C
--- Negotiation Round 1 ---
ACME offer: $3000.00
A counter: $3200.00
C counter: $3100.00
...
```

## Environment

Create `.env` file:
```
OPENAI_API_KEY=sk-...
LLM_CONFIG=default
```

## Requirements

- Python 3.9+
- `openai` — For OpenAI API (gpt-5-nano, gpt-4-turbo)
- `mcp` — For MCP server/client
- `pyyaml` — For contractor config
- `python-dotenv` — For .env loading
