# MCP House Building - System Architecture

## Overview

A multi-agent construction contracting simulation system using MCP-style servers and LLM-driven agents. ACME (buyer) procures construction work from companies A-F via:
1. **Reverse Dutch auctions** (prices increase, max 3 rounds)
2. **Monotonic concession negotiation** (alternating offers until agreement)

## Core Principles

1. **Servers are source of truth** - maintain all state
2. **Agents are LLMs** - make decisions by discovering and calling tools
3. **Tool discovery, not imposition** - prompts guide LLMs to discover MCP tools
4. **Workflows are code-driven** - simulator controls flow, LLM makes decisions at each step
5. **Full replayability** - all actions logged with before/after state

## Architecture

### Servers (Stateful Simulation Engines)

#### AuctionServer (`auction_server/`)
- **State**: Current auction for a task, rounds (0-3), bids, agent tracking
- **Tools**: 
  - `propose_budget(price)` - ACME only
  - `bid()` - Companies only
  - `get_status()` - Everyone
- **Protocol**: Fixed 3 rounds, companies bid YES/NO at ACME's proposed prices

**Key Files:**
- `server.py` - Main server, tool routing by agent_name
- `state.py` - AuctionState dataclass
- `tools.py` - AuctionTools implementation

#### NegotiationServer (`negotiation_server/`)
- **State**: Task, bidders, offers history, agreement status
- **Tools**:
  - `make_offer(price, type)` - ACME and companies
  - `get_status()` - Everyone
- **Protocol**: Up to 3 rounds, monotonic concession (each offer moves toward agreement)

**Key Files:**
- `server.py` - Main server, tool routing
- `state.py` - NegotiationState dataclass
- `tools.py` - NegotiationTools implementation

### Agents (LLM-Driven Decision Policies)

#### BaseAgent (`agents/base_agent.py`)
- **Client**: OpenAI client (gpt-5-nano default)
- **Memory**: Minimal (last_action, last_offer_seen)
- **LLM Config**: Via `config/llm_config.py`
- **Methods**:
  - `think(prompt)` - Call LLM, extract response
  - `decide(prompt)` - Parse JSON decision
  - `register(auction_server, negotiation_server)` - Register with servers

#### ACMEAgent (`agents/acme.py`)
Buyer agent with two code-driven workflows:

**Auction Workflow** (`run_auction_workflow(tasks)`)
- Code loop: for each task
- Code: start auction session
- Code loop: for 3 rounds
  - Get status + available tools
  - **LLM decides price to propose** (via `acme_auction_round.txt` prompt)
  - Code: execute propose_budget tool
  - Code: check results
- Code: extract winner

**Negotiation Workflow** (`run_negotiation_workflow(auction_winners, tasks)`)
- Code loop: for each task-winner pair
- Code: start negotiation session
- Code loop: up to 3 rounds
  - Get status + available tools
  - **LLM decides offer (price + type)** (via `acme_negotiation_round.txt` prompt)
  - Code: execute make_offer tool
  - Code: check if agreed
- Code: extract final price

#### CompanyAgent (`agents/company.py`)
Contractor agent with two workflows:

**Auction Workflow** (`run_auction_workflow(tasks)`)
- Code loop: for 3 rounds (listens to auction)
  - Get status + available tools
  - **LLM decides to bid or skip** (via `company_auction_round.txt` prompt)
  - Code: execute bid tool if decided
- Code: return won task or empty string

**Negotiation Workflow** (`run_negotiation_workflow(task_name, task_budget)`)
- Code loop: up to 3 rounds
  - Get status + available tools
  - **LLM decides offer (price + type)** (via `company_negotiation_round.txt` prompt)
  - Code: execute make_offer tool
  - Code: check if agreed
- Code: return final price + agreement status

### Prompts (LLM Guidance)

Each agent has round-specific prompts that:
1. Show available tools (from `get_available_tools()`)
2. Explain current state
3. Ask LLM to discover and use the right tool
4. Format response as JSON with `{"tool": "...", "arguments": {...}}`

**Key Prompts:**
- `acme_auction_round.txt` - ACME decides price for round N
- `company_auction_round.txt` - Company decides whether to bid
- `acme_negotiation_round.txt` - ACME decides offer (price + type)
- `company_negotiation_round.txt` - Company decides offer (price + type)

### Configuration

#### LLM Config (`config/llm_config.py`)
- **Default**: gpt-5-nano with medium reasoning
- **Alternatives**: fast (gpt-4-turbo), deep (gpt-5-nano + high reasoning)
- **Override**: `LLM_CONFIG=<config_name>` environment variable

#### Task Config (`config/acme_config.py`)
- 4 construction tasks with budgets
- Total budget: $24,000

#### Contractor Config (`config/contractors.yaml`)
- 6 companies (A-F) with specialties and costs
- Each company can bid on multiple tasks

### Simulation Runner (`run_simulation.py`)

Main orchestrator that:
1. Loads OpenAI client from `.env` (OPENAI_API_KEY)
2. Initializes both MCP servers
3. Creates ACME agent + 6 company agents
4. For each task:
   - Calls `acme_agent.run_auction_workflow([task])`
   - If winner, calls `acme_agent.run_negotiation_workflow(winners, [task])`
   - For each company, calls their auction + negotiation workflows in sync
5. Logs all actions with before/after state
6. Prints summary (budget, cost, savings)

### Shared Utilities

#### Types (`shared/types.py`)
- Task, Specialty, Company, Offer, Bid, AgentState dataclasses
- Phase, OfferType enums

#### Logger (`shared/logger.py`)
- StructuredLogger: logs actions as JSON with state snapshots
- Enables full simulation replay

#### Utils (`shared/utils.py`)
- `safe_json_parse()` - Extract JSON from LLM response

## Execution Flow

### Auction Phase (per task)

```
1. ACME calls run_auction_workflow()
2.   server.start_auction(task)
3.   for round in 1..3:
4.     ACME: think() → decide price
5.     server.propose_budget(acme_price)
6.     for company in companies:
7.       Company: think() → decide bid/skip
8.       if bid: server.bid()
9.     announce round bids
10.  return winner
```

**Key**: Rounds 1-3 run to completion; companies decide each round whether to bid.

### Negotiation Phase (per task-winner pair)

```
1. ACME calls run_negotiation_workflow()
2.   server.start_negotiation(task, [winner])
3.   for round in 1..3:
4.     ACME: think() → decide offer (price + type)
5.     server.make_offer(price, type)
6.     if type == "accept": agreed = true; break
7.     Company: think() → decide offer (price + type)
8.     server.make_offer(price, type)
9.     if type == "accept": agreed = true; break
10.  return agreed + final_price
```

**Key**: Alternating offers, monotonic concession (each offer moves toward agreement).

## Development Setup

1. Copy `.env.example` to `.env`
2. Add your OpenAI API key: `OPENAI_API_KEY=sk-...`
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python run_simulation.py`

### Configuration Options

```bash
# Use default config (gpt-5-nano + medium reasoning)
python run_simulation.py

# Use fast config (gpt-4-turbo)
LLM_CONFIG=fast python run_simulation.py

# Use deep reasoning
LLM_CONFIG=deep python run_simulation.py
```

## Key Design Decisions

1. **Code-driven workflows with LLM decisions**
   - Simulator controls loop structure
   - LLM decides tool calls at each step
   - Allows intermediate analysis and adaptation

2. **Tool discovery via prompts**
   - Agents receive tool descriptions
   - Prompts ask LLM to pick the right tool
   - No hardcoded tool names in agent logic

3. **Minimal agent memory**
   - Only `last_action` and `last_offer_seen`
   - Server is source of truth for all state
   - Prevents hallucination and inconsistency

4. **Fixed protocol enforcement**
   - Server enforces 3 auction rounds
   - Server enforces up to 3 negotiation rounds
   - Server enforces monotonic concession rules
   - Agents can't bypass or circumvent protocol

5. **Full auditability**
   - Every action logged with state snapshots
   - Enables replay and debugging
   - Shows exactly what LLM decided at each step

## Testing Checklist

- [ ] Auction runs 3 rounds with multiple companies bidding
- [ ] Companies only bid if price >= their cost
- [ ] Negotiation alternates ACME ↔ Company offers
- [ ] Agreements respect monotonic concession
- [ ] Final prices are ≤ budget
- [ ] All actions logged with state before/after
- [ ] Simulation completes for all 4 tasks
- [ ] Summary shows budget vs. actual cost

## Future Enhancements

- Multiple concurrent negotiations per task
- Task-dependent company specialization
- Reputation-based bidding (history across tasks)
- Parallel auctions instead of sequential
- Web UI for live monitoring
