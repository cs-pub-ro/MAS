import json
from typing import Dict, Any, List
from agents.base_agent import BaseAgent


class ACMEAgent(BaseAgent):
    """ACME buyer agent - runs workflows that call LLM at each step."""

    def __init__(self, client):
        super().__init__("ACME", client)
        self.prompt_dir = "prompts"

    def _load_prompt(self, path: str) -> str:
        """Load prompt from file."""
        with open(path) as f:
            return f.read()

    def run_auction_workflow(self, tasks: list) -> Dict[str, List[str]]:
        """
        ACME auction workflow - code-driven with LLM decisions at each step.

        REVERSE DUTCH AUCTION: All companies that can do the work at the offered price
        submit bids. ACME then negotiates with ALL bidders.

        Workflow:
        1. For each task:
           a. Start auction session
           b. For 3 rounds:
              - Get current status (all bids so far)
              - LLM decides what price to propose (tool discovery)
              - Execute price proposal
              - Companies bid YES/NO at this price
           c. Collect ALL companies that said YES
           d. Return all bidders (not just lowest)

        Returns: {task_name: [list of companies that bid YES]}
        """
        bidders_by_task = {}

        for task in tasks:
            self.update_memory(f"auction_started:{task.name}")

            # Step 1: Start auction session (direct call, not a tool)
            self.auction_server.start_auction(task)

            # Step 2: Run 3 auction rounds
            for round_num in range(1, 4):
                # Get available tools and current status
                available_tools = self.auction_server.get_available_tools(self.name)
                current_status = self.auction_server.execute_tool(
                    self.name, "get_status", {}
                )

                # Load prompt for this round decision
                prompt = self._load_prompt("prompts/acme_auction_round.txt")
                prompt = prompt.format(
                    task_name=task.name,
                    budget=task.budget,
                    round=round_num,
                    current_price=current_status.get("current_price", 0),
                    current_bids=json.dumps(current_status.get("bids", [])),
                    available_tools=json.dumps(available_tools, indent=2),
                )

                # LLM decides what tool to call
                response = self.think(prompt)

                try:
                    decision = json.loads(response)
                    tool_name = decision.get("tool")
                    tool_args = decision.get("arguments", {})

                    # Execute the tool LLM decided on
                    if tool_name == "propose_budget":
                        result = self.auction_server.execute_tool(
                            self.name, tool_name, tool_args
                        )
                        self.update_memory(f"round_{round_num}:{tool_args.get('price')}")
                except Exception as e:
                    print(f"Error in auction round {round_num}: {e}")

            # Step 3: Get final status and collect ALL bidders
            final_status = self.auction_server.execute_tool(self.name, "get_status", {})
            bids = final_status.get("bids", [])
            bidders = [bid["company"] for bid in bids] if bids else []

            if bidders:
                bidders_by_task[task.name] = bidders
                self.update_memory(f"auction_won:{task.name}:{','.join(bidders)}")
            else:
                self.update_memory(f"auction_failed:{task.name}")

        return bidders_by_task

    def run_negotiation_workflow(
        self, auction_bidders: Dict[str, List[str]], tasks: list
    ) -> Dict[str, Dict[str, Any]]:
        """
        ACME negotiation workflow with MULTIPLE CONCURRENT sessions per task.

        ACME negotiates with ALL companies that said YES in the auction.
        For each round:
        1. ACME makes offers to ALL bidders for ALL tasks simultaneously
        2. ACME waits for ALL companies to respond
        3. ACME analyzes all responses
        4. ACME makes next round of offers (informed by all responses)

        This gives ACME strategic advantage: it sees all companies' responses
        before deciding next move for each.

        Returns: {task_name: {contractor, final_price, agreed}}
        """
        results = {}

        # Step 1: Start parallel negotiation sessions for all (task, bidder) pairs
        negotiations = {}  # {(task_name, company): None} - tracks all sessions
        for task in tasks:
            bidders = auction_bidders.get(task.name, [])
            if not bidders:
                continue
            # Start ONE negotiation session per task with ALL bidders
            self.negotiation_server.start_negotiation(task, bidders)
            self.update_memory(f"negotiation_started:{task.name}:{','.join(bidders)}")
            for bidder in bidders:
                negotiations[(task.name, bidder)] = None

        # Step 2: Run negotiation rounds (up to 3) for ALL tasks/bidders in parallel
        for round_num in range(1, 4):
            # ACME Round: Make offers to all bidders for all tasks
            for task_name, _ in negotiations.items():
                if not task_name:
                    continue

                # Find task object
                task = next((t for t in tasks if t.name == task_name), None)
                if not task:
                    continue

                # Set current task for this negotiation session
                self.negotiation_server.set_current_task(task_name)

                # Get available tools and current status
                available_tools = self.negotiation_server.get_available_tools(self.name)
                current_status = self.negotiation_server.execute_tool(
                    self.name, "get_status", {}
                )

                # Check if all bidders agreed
                if current_status.get("agreed_contractor"):
                    self.update_memory(f"task_agreed:{task_name}")
                    continue

                bidders = auction_bidders.get(task_name, [])
                bidders_str = ", ".join(bidders)

                # Load prompt for this round decision
                prompt = self._load_prompt("prompts/acme_negotiation_round.txt")
                prompt = prompt.format(
                    task_name=task_name,
                    company=bidders_str,
                    budget=task.budget,
                    round=round_num,
                    offers_history=json.dumps(current_status.get("offers", []), indent=2),
                    available_tools=json.dumps(available_tools, indent=2),
                )

                # LLM decides what tool to call
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
                        self.update_memory(
                            f"round_{round_num}:{task_name}:{tool_args.get('price')}"
                        )
                except Exception as e:
                    print(f"Error in ACME negotiation round {round_num} for {task_name}: {e}")

            # Companies Round: They respond (simulator calls their run_negotiation_workflow)
            # ACME waits here for all responses before next round

        # Step 3: Get final status for all task negotiations
        for task_name in set([key[0] for key in negotiations.keys()]):
            # Set current task
            self.negotiation_server.set_current_task(task_name)
            final_status = self.negotiation_server.execute_tool(
                self.name, "get_status", {}
            )
            final_price = final_status.get("agreed_price", 0)
            contractor = final_status.get("agreed_contractor", "")

            results[task_name] = {
                "contractor": contractor,
                "final_price": final_price,
                "agreed": bool(contractor),
            }

            self.update_memory(
                f"negotiation_complete:{task_name}:{contractor}:{final_price}"
                if contractor
                else f"negotiation_failed:{task_name}"
            )

        return results
