"""Orchestrator for MCP-based multi-agent simulation.

Connects to FastMCP servers via SSE HTTP transport and orchestrates
the auction and negotiation phases.
"""

import json
import yaml
import os
import asyncio
from typing import Dict, Any, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

from config.acme_config import ACME_TASKS
from agents.acme import ACMEAgent
from agents.company import CompanyAgent
from shared.types import Specialty
from shared.logger import StructuredLogger


class Orchestrator:
    """Orchestrates auction and negotiation via FastMCP servers."""

    def __init__(self, auction_session, negotiation_session):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.logger = StructuredLogger("Orchestrator")

        with open("config/contractors.yaml") as f:
            self.contractors_config = yaml.safe_load(f)

        self.company_names = [c["name"] for c in self.contractors_config["companies"]]
        self.companies_data = {
            c["name"]: c for c in self.contractors_config["companies"]
        }

        # Store MCP sessions
        self.auction_session = auction_session
        self.negotiation_session = negotiation_session

        # Create agents
        self.acme_agent = ACMEAgent(self.client)
        self.company_agents: Dict[str, CompanyAgent] = {}

        self._init_company_agents()

    def _init_company_agents(self):
        """Initialize company agents from config."""
        for name in self.company_names:
            company_config = self.companies_data[name]
            specialties = [
                Specialty(
                    specialty=s["specialty"],
                    cost=s["cost"],
                )
                for s in company_config["specialties"]
            ]
            self.company_agents[name] = CompanyAgent(
                name=name,
                specialties=specialties,
                client=self.client,
            )

    async def _get_auction_tools(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get tools from auction server."""
        result = await self.auction_session.list_tools()
        tools = result.tools if hasattr(result, 'tools') else result
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.inputSchema,
            }
            for t in tools
        ]

    async def _call_auction_tool(self, tool_name: str, arguments: Dict[str, Any], task_name: str = None) -> Dict[str, Any]:
        """Call tool on auction server."""
        # For start_auction, task_name is already in arguments
        # For other tools, add it explicitly
        if task_name and tool_name in ["propose_budget", "bid", "get_status"]:
            arguments = {**arguments, "task_name": task_name}
        self.logger.log_info(f"Calling {tool_name} with args: {arguments}")
        result = await self.auction_session.call_tool(tool_name, arguments)
        if result.content and len(result.content) > 0:
            return json.loads(result.content[0].text)
        return {"status": "error", "message": "No result"}

    async def _get_negotiation_tools(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get tools from negotiation server."""
        result = await self.negotiation_session.list_tools()
        tools = result.tools if hasattr(result, 'tools') else result
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.inputSchema,
            }
            for t in tools
        ]

    async def _call_negotiation_tool(self, tool_name: str, arguments: Dict[str, Any], task_name: str = None) -> Dict[str, Any]:
        """Call tool on negotiation server."""
        # For start_negotiation, task_name is already in arguments
        # For other tools, add it explicitly
        if task_name and tool_name in ["make_offer", "get_status"]:
            arguments = {**arguments, "task_name": task_name}
        result = await self.negotiation_session.call_tool(tool_name, arguments)
        if result.content and len(result.content) > 0:
            return json.loads(result.content[0].text)
        return {"status": "error", "message": "No result"}

    async def run_auction(self, task) -> List[str]:
        """Run auction phase for a task."""
        self.logger.log_info(f"\n{'='*60}")
        self.logger.log_info(f"AUCTION PHASE: {task.name}")
        self.logger.log_info(f"Budget: ${task.budget:.2f}")
        self.logger.log_info(f"{'='*60}")

        # Initialize auction
        await self._call_auction_tool("start_auction", {
            "task_name": task.name,
            "budget": task.budget,
        })

        # Run 3 rounds
        for round_num in range(1, 4):
            self.logger.log_info(f"\n--- Round {round_num} ---")

            # ACME proposes price
            available_tools = await self._get_auction_tools("ACME")
            current_status = await self._call_auction_tool("get_status", {}, task.name)

            prompt = self.acme_agent._load_prompt("prompts/acme_auction_round.txt")
            prompt = prompt.format(
                task_name=task.name,
                budget=task.budget,
                round=round_num,
                current_price=current_status.get("current_price", 0),
                current_bids=json.dumps(current_status.get("bids", []), indent=2),
                available_tools=json.dumps(available_tools, indent=2),
            )

            response = self.acme_agent.think(prompt)
            try:
                decision = json.loads(response)
                tool_name = decision.get("tool")
                tool_args = decision.get("arguments", {})

                if tool_name == "propose_budget":
                    result = await self._call_auction_tool(tool_name, tool_args, task.name)
                    price = tool_args.get("price", 0)
                    self.logger.log_info(f"ACME proposes: ${price:.2f}")
                    if result.get("status") != "success":
                        self.logger.log_info(f"  Error: {result.get('message')}")
            except Exception as e:
                self.logger.log_info(f"Error in ACME decision: {e}")

            # Companies decide whether to bid
            for company_name in self.company_names:
                available_tools = await self._get_auction_tools(company_name)
                current_status = await self._call_auction_tool("get_status", {}, task.name)

                cost = self.company_agents[company_name].get_cost_for_task(task.name)
                if cost is None:
                    continue

                current_price = current_status.get("current_price", 0)

                prompt = self.company_agents[company_name]._load_prompt(
                    "prompts/company_auction_round.txt"
                )
                prompt = prompt.format(
                    company_name=company_name,
                    task_name=task.name,
                    round=round_num,
                    current_price=current_price,
                    company_cost=cost,
                    profit=current_price - cost if current_price >= cost else 0,
                    specialties="\n".join(
                        f"- {s.specialty}: ${s.cost}"
                        for s in self.company_agents[company_name].specialties
                    ),
                    available_tools=json.dumps(available_tools, indent=2),
                )

                response = self.company_agents[company_name].think(prompt)
                try:
                    decision = json.loads(response)
                    tool_name = decision.get("tool")

                    if tool_name == "bid":
                        result = await self._call_auction_tool("bid", {"company": company_name}, task.name)
                        if result.get("status") == "success":
                            self.logger.log_info(f"  {company_name}: Bid submitted at ${current_price:.2f}")
                        else:
                            self.logger.log_info(f"  {company_name}: Bid failed - {result.get('message', 'unknown error')}")
                    else:
                        self.logger.log_info(f"  {company_name}: Passed (decided not to bid)")
                except Exception as e:
                    self.logger.log_info(f"  {company_name}: Error parsing decision: {e}")
                    self.logger.log_info(f"    Raw response: {response[:200]}")

        # Get final bidders
        final_status = await self._call_auction_tool("get_status", {}, task.name)
        all_bids = final_status.get("bids", [])
        bidders = list(set(bid["company"] for bid in all_bids))

        self.logger.log_info(f"\nAuction complete. Bidders: {bidders}")
        return bidders

    async def run_negotiation(self, task, bidders: List[str]) -> Dict[str, Any]:
        """Run negotiation phase with all auction bidders."""
        if not bidders:
            self.logger.log_info("No bidders, skipping negotiation")
            return {}

        self.logger.log_info(f"\n{'='*60}")
        self.logger.log_info(f"NEGOTIATION PHASE: {task.name}")
        self.logger.log_info(f"Bidders: {', '.join(bidders)}")
        self.logger.log_info(f"{'='*60}")

        # Initialize negotiation
        await self._call_negotiation_tool("start_negotiation", {
            "task_name": task.name,
            "budget": task.budget,
            "bidders": bidders,
        }, None)

        # Run up to 3 rounds
        for round_num in range(1, 4):
            self.logger.log_info(f"\n--- Round {round_num} ---")

            # ACME makes offer
            available_tools = await self._get_negotiation_tools("ACME")
            current_status = await self._call_negotiation_tool("get_status", {}, task.name)

            if current_status.get("agreed_contractor"):
                self.logger.log_info("Agreement already reached")
                break

            prompt = self.acme_agent._load_prompt("prompts/acme_negotiation_round.txt")
            bidders_str = ", ".join(bidders)
            prompt = prompt.format(
                task_name=task.name,
                company=bidders_str,
                budget=task.budget,
                round=round_num,
                offers_history=json.dumps(current_status.get("offers", []), indent=2),
                available_tools=json.dumps(available_tools, indent=2),
            )

            response = self.acme_agent.think(prompt)
            try:
                decision = json.loads(response)
                tool_name = decision.get("tool")
                tool_args = decision.get("arguments", {})

                if tool_name == "make_offer":
                    tool_args["from_"] = "ACME"
                    tool_args["to_"] = bidders[0] if bidders else "ACME"
                    result = await self._call_negotiation_tool(tool_name, tool_args, task.name)
                    price = tool_args.get("price", 0)
                    offer_type = tool_args.get("type", "offer")
                    self.logger.log_info(f"ACME {offer_type}: ${price:.2f}")
                    if result.get("agreement_reached"):
                        break
            except Exception as e:
                self.logger.log_info(f"Error in ACME negotiation: {e}")

            # Companies respond
            for company_name in bidders:
                available_tools = await self._get_negotiation_tools(company_name)
                current_status = await self._call_negotiation_tool("get_status", {}, task.name)

                if current_status.get("agreed_contractor"):
                    break

                cost = self.company_agents[company_name].get_cost_for_task(task.name)
                if cost is None:
                    cost = self.company_agents[company_name].specialties[0].cost

                prompt = self.company_agents[company_name]._load_prompt(
                    "prompts/company_negotiation_round.txt"
                )
                num_competitors = len(bidders) - 1
                prompt = prompt.format(
                    company_name=company_name,
                    task_name=task.name,
                    round=round_num,
                    company_cost=cost,
                    task_budget=task.budget,
                    acme_last_offer=0,
                    num_competitors=num_competitors,
                    offers_history=json.dumps(current_status.get("offers", []), indent=2),
                    available_tools=json.dumps(available_tools, indent=2),
                )

                response = self.company_agents[company_name].think(prompt)
                try:
                    decision = json.loads(response)
                    tool_name = decision.get("tool")
                    tool_args = decision.get("arguments", {})

                    if tool_name == "make_offer":
                        tool_args["from_"] = company_name
                        tool_args["to_"] = "ACME"
                        result = await self._call_negotiation_tool(tool_name, tool_args, task.name)
                        price = tool_args.get("price", 0)
                        offer_type = tool_args.get("type", "offer")
                        self.logger.log_info(f"  {company_name} {offer_type}: ${price:.2f}")
                        if result.get("agreement_reached"):
                            break
                except Exception as e:
                    self.logger.log_info(f"  {company_name}: Error - {e}")

        # Get final status
        final_status = await self._call_negotiation_tool("get_status", {}, task.name)
        return {
            "contractor": final_status.get("agreed_contractor", ""),
            "final_price": final_status.get("agreed_price", 0),
            "agreed": bool(final_status.get("agreed_contractor")),
        }

    async def run(self):
        """Run complete simulation."""
        self.logger.log_info("\n" + "="*60)
        self.logger.log_info("MCP HOUSE BUILDING SIMULATION")
        self.logger.log_info("="*60)
        self.logger.log_info(f"Tasks: {len(ACME_TASKS)}")
        self.logger.log_info(f"Companies: {', '.join(self.company_names)}")

        contracts = {}

        for task in ACME_TASKS:
            bidders = await self.run_auction(task)

            if bidders:
                negotiation_result = await self.run_negotiation(task, bidders)
                if negotiation_result:
                    contracts[task.name] = negotiation_result

        # Summary
        self.logger.log_info(f"\n{'='*60}")
        self.logger.log_info("SIMULATION SUMMARY")
        self.logger.log_info(f"{'='*60}")

        total_budget = sum(t.budget for t in ACME_TASKS)
        total_cost = sum(
            c["final_price"]
            for c in contracts.values()
            if c.get("agreed", False)
        )
        savings = total_budget - total_cost

        for task_name, contract in contracts.items():
            if contract.get("agreed"):
                self.logger.log_info(
                    f"{task_name}: {contract['contractor']} @ ${contract['final_price']:.2f}"
                )
            else:
                self.logger.log_info(f"{task_name}: No agreement reached")

        self.logger.log_info(f"\nTotal Budget:  ${total_budget:.2f}")
        self.logger.log_info(f"Total Cost:    ${total_cost:.2f}")
        self.logger.log_info(f"Savings:       ${savings:.2f}")


if __name__ == "__main__":
    print("="*60, flush=True)
    print("MCP House Building Simulation", flush=True)
    print("="*60, flush=True)
    print("", flush=True)

    async def main():
        try:
            # Connect to FastMCP servers via SSE
            async with sse_client("http://localhost:8010/sse") as (auction_read, auction_write):
                async with sse_client("http://localhost:8011/sse") as (negotiation_read, negotiation_write):
                    async with ClientSession(auction_read, auction_write) as auction_session:
                        async with ClientSession(negotiation_read, negotiation_write) as negotiation_session:
                            # Initialize MCP handshake
                            await auction_session.initialize()
                            await negotiation_session.initialize()

                            print("✓ Connected to servers", flush=True)
                            print("", flush=True)

                            orchestrator = Orchestrator(auction_session, negotiation_session)
                            await orchestrator.run()

            print("\n✓ Simulation complete.", flush=True)
        except Exception as e:
            print(f"\n✗ Error: {e}", flush=True)
            raise

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✗ Stopped by user.", flush=True)
    except Exception:
        raise
