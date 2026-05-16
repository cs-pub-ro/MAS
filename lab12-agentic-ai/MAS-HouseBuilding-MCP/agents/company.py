import json
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from shared.types import Specialty


class CompanyAgent(BaseAgent):
    """Company agent - runs workflows that call LLM at each step."""

    def __init__(self, name: str, specialties: List[Specialty], client):
        super().__init__(name, client)
        self.specialties = specialties
        self.prompt_dir = "prompts"

    def _load_prompt(self, path: str) -> str:
        """Load prompt from file."""
        with open(path) as f:
            return f.read()

    def get_cost_for_task(self, task_name: str) -> float:
        """Get cost for a specific task, or None if not applicable."""
        spec = self._get_specialty_for_task(task_name)
        return spec.cost if spec else None

    def _get_specialty_for_task(self, task_name: str) -> Specialty:
        """Find specialty matching task name."""
        for spec in self.specialties:
            if spec.specialty.lower() in task_name.lower():
                return spec
        return None

    def run_auction_workflow(self, tasks: list):
        """
        Company auction workflow - code-driven with LLM decisions at each step.

        Workflow:
        1. Listen to each auction round
        2. For each round:
           a. Get current status
           b. LLM decides whether to bid (tool discovery)
           c. Execute bid or skip
           d. Check if won
        """
        won_task = None

        # This runs during the auction - company listens to all rounds
        for round_num in range(1, 4):
            # Get available tools and current status
            available_tools = self.auction_server.get_available_tools(self.name)
            current_status = self.auction_server.execute_tool(self.name, "get_status", {})

            current_task = current_status.get("task")
            current_price = current_status.get("current_price", 0)

            if not current_task:
                break

            # Check if we can do this task
            cost = self.get_cost_for_task(current_task)
            if cost is None:
                continue

            # Load prompt for this round decision
            prompt = self._load_prompt("prompts/company_auction_round.txt")
            prompt = prompt.format(
                company_name=self.name,
                task_name=current_task,
                round=round_num,
                current_price=current_price,
                company_cost=cost,
                profit=current_price - cost if current_price >= cost else -(cost - current_price),
                specialties="\n".join(f"- {s.specialty}: ${s.cost}" for s in self.specialties),
                available_tools=json.dumps(available_tools, indent=2),
            )

            # LLM decides whether to bid
            response = self.think(prompt)

            try:
                decision = json.loads(response)
                tool_name = decision.get("tool")

                # Execute the tool LLM decided on
                if tool_name == "bid":
                    result = self.auction_server.execute_tool(self.name, "bid", {})
                    if result.get("status") == "success":
                        self.update_memory(f"bid:{current_task}:{current_price}")
                        won_task = current_task
            except Exception as e:
                print(f"Error in {self.name} auction round {round_num}: {e}")

        return won_task

    def run_negotiation_workflow(self, task_name: str, task_budget: float):
        """
        Company negotiation workflow - code-driven with LLM decisions at each step.

        Workflow:
        1. Company listens to ACME's offers
        2. For up to 3 rounds:
           a. Get current status and ACME's latest offer
           b. LLM decides how to respond (tool discovery)
           c. Execute response
           d. Check if agreed
        """
        cost = self.get_cost_for_task(task_name)
        if cost is None:
            cost = self.specialties[0].cost

        self.update_memory(f"negotiation_started:{task_name}")

        agreed = False

        for round_num in range(1, 4):
            # Get available tools and current status
            available_tools = self.negotiation_server.get_available_tools(self.name)
            current_status = self.negotiation_server.execute_tool(
                self.name, "get_status", {}
            )

            offers = current_status.get("offers", [])

            # Find ACME's latest offer
            acme_last_offer = 0
            for offer in reversed(offers):
                if offer.get("from") == "ACME":
                    acme_last_offer = offer.get("price", 0)
                    break

            # Load prompt for this round decision
            prompt = self._load_prompt("prompts/company_negotiation_round.txt")
            prompt = prompt.format(
                company_name=self.name,
                task_name=task_name,
                round=round_num,
                company_cost=cost,
                task_budget=task_budget,
                acme_last_offer=acme_last_offer,
                offers_history=json.dumps(offers, indent=2),
                available_tools=json.dumps(available_tools, indent=2),
            )

            # LLM decides what to offer
            response = self.think(prompt)

            try:
                decision = json.loads(response)
                tool_name = decision.get("tool")
                tool_args = decision.get("arguments", {})

                # Execute the tool LLM decided on
                if tool_name == "make_offer":
                    result = self.negotiation_server.execute_tool(
                        self.name, tool_name, tool_args
                    )
                    if result.get("agreement_reached"):
                        agreed = True
                        break
                    self.update_memory(
                        f"round_{round_num}:{tool_args.get('price')}"
                    )
            except Exception as e:
                print(f"Error in {self.name} negotiation round {round_num}: {e}")

        # Get final status
        final_status = self.negotiation_server.execute_tool(self.name, "get_status", {})
        final_price = final_status.get("agreed_price", cost)

        return {
            "contractor": self.name,
            "final_price": final_price,
            "agreed": agreed,
        }
