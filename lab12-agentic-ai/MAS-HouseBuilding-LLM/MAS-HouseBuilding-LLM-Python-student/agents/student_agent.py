"""
Student agents — reasoning is delegated to an OpenAI LLM via prompts/.

The decision-making methods (`propose_item_budget`, `provide_negotiation_offer`,
`decide_bid`, `respond_to_offer`) load their prompt from ``prompts/<name>.txt``,
substitute ``$placeholders`` with the function arguments + agent state, and
call OpenAI through ``llm_client.call_llm``.

Students normally edit only the files in ``prompts/``; they should not need
to change this file.

The ``notify_*`` methods do not call the LLM — they record game facts in
Python state and expose them to the next decision prompt as placeholders
(``$auction_history``, ``$negotiation_history``, ``$current_contracts``, ...).
"""
from typing import Any, Dict, List
import logging

from agents import HouseOwnerAgent, CompanyAgent
from communication import NegotiationMessage
from llm_client import call_llm

logger = logging.getLogger("agents")


class MyACMEAgent(HouseOwnerAgent):

    def __init__(self, role: str, budget_list: List[Dict[str, Any]]):
        super().__init__(role, budget_list)
        # Per item: list of {round, proposed_budget, responding_agents}
        self._auction_history: Dict[str, List[Dict[str, Any]]] = {}
        # (item, round) -> last proposed budget, so notify_auction_round_result can backfill it
        self._last_proposed_budget: Dict[tuple, float] = {}
        # Per item, per partner: list of {round, my_offer, partner_response}
        self._negotiations: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        # Per item: {winner, offer}
        self._negotiation_outcomes: Dict[str, Dict[str, Any]] = {}

    def _log(self, msg: str, *args: Any) -> None:
        logger.info("[%s] " + msg, self.name, *args)

    # ---- decisions (LLM-backed) ----

    def propose_item_budget(self, auction_item: str, auction_round: int) -> float:
        max_budget = self.budget_dict[auction_item]
        ctx = {
            "role":             self.role,
            "auction_item":     auction_item,
            "auction_round":    auction_round,
            "max_budget":       max_budget,
            "all_budgets":      self.budget_dict,
            "auction_history":  self._auction_history.get(auction_item, []),
        }
        budget = call_llm(
            prompt_name="acme_propose_item_budget",
            agent_name=self.role,
            function_name="propose_item_budget",
            context=ctx,
            expect="float",
        )
        self._last_proposed_budget[(auction_item, auction_round)] = budget
        return budget

    def provide_negotiation_offer(self, negotiation_item: str, partner_agent: str,
                                  negotiation_round: int) -> float:
        history = self._negotiations.get(negotiation_item, {}).get(partner_agent, [])
        my_prev       = history[-1]["my_offer"]          if history else None
        partner_last  = history[-1]["partner_response"]  if history else None

        # Recover the price ACME actually paid in the auction (the last successful round)
        auction_price = None
        for record in self._auction_history.get(negotiation_item, []):
            if record.get("responding_agents"):
                auction_price = record["proposed_budget"]
        max_budget = self.budget_dict[negotiation_item]

        ctx = {
            "role":                  self.role,
            "negotiation_item":      negotiation_item,
            "partner_agent":         partner_agent,
            "negotiation_round":     negotiation_round,
            "auction_price":         auction_price,
            "max_budget":            max_budget,
            "my_previous_offer":     my_prev,
            "partner_last_response": partner_last,
            "negotiation_history":   history,
        }
        offer = call_llm(
            prompt_name="acme_provide_negotiation_offer",
            agent_name=self.role,
            function_name="provide_negotiation_offer",
            context=ctx,
            expect="float",
        )

        self._negotiations.setdefault(negotiation_item, {}) \
                          .setdefault(partner_agent, []) \
                          .append({
            "round": negotiation_round,
            "my_offer": offer,
            "partner_response": None,
        })
        return offer

    # ---- notifications (state-only, no LLM) ----

    def notify_auction_round_result(self, auction_item: str, auction_round: int,
                                    responding_agents: List[str]) -> None:
        proposed = self._last_proposed_budget.get((auction_item, auction_round))
        self._auction_history.setdefault(auction_item, []).append({
            "round":              auction_round,
            "proposed_budget":    proposed,
            "responding_agents":  list(responding_agents),
        })
        self._log("auction round %d for %s -> responding=%s",
                  auction_round, auction_item, responding_agents)

    def notify_partner_response(self, response_msg: NegotiationMessage) -> None:
        item    = response_msg.negotiation_item
        partner = response_msg.sender
        rnd     = response_msg.round

        history = self._negotiations.setdefault(item, {}).setdefault(partner, [])
        for entry in history:
            if entry["round"] == rnd:
                entry["partner_response"] = response_msg.offer
                break
        else:
            # First-round flow: env asks for partner response before we recorded our offer
            history.append({
                "round": rnd,
                "my_offer": None,
                "partner_response": response_msg.offer,
            })
        self._log("partner %s responded %.2f for %s in round %d",
                  partner, response_msg.offer, item, rnd)

    def notify_negotiation_winner(self, negotiation_item: str, winning_agent: str,
                                  winning_offer: float) -> None:
        self._negotiation_outcomes[negotiation_item] = {
            "winner": winning_agent,
            "offer":  winning_offer,
        }
        self._log("negotiation winner for %s: %s @ %.2f",
                  negotiation_item, winning_agent, winning_offer)


class MyCompanyAgent(CompanyAgent):

    def __init__(self, role: str, specialties: List[Dict[str, Any]]):
        super().__init__(role, specialties)
        # Per item: {round, num_selected}
        self._won_auctions: Dict[str, Dict[str, Any]] = {}
        # Per item: list of {round, initiator_offer, my_response}
        self._negotiation_history: Dict[str, List[Dict[str, Any]]] = {}
        # item -> final price
        self._contracts: Dict[str, float] = {}
        self._lost_negotiations: List[str] = []

    def _log(self, msg: str, *args: Any) -> None:
        logger.info("[%s] " + msg, self.name, *args)

    # ---- decisions (LLM-backed) ----

    def decide_bid(self, auction_item: str, auction_round: int, item_budget: float) -> bool:
        my_cost = self.specialties.get(auction_item)
        ctx = {
            "role":               self.role,
            "auction_item":       auction_item,
            "auction_round":      auction_round,
            "item_budget":        item_budget,
            "my_cost":            my_cost,
            "my_specialties":     self.specialties,
            "current_contracts":  list(self._contracts.keys()),
            "lost_negotiations":  list(self._lost_negotiations),
        }
        return call_llm(
            prompt_name="company_decide_bid",
            agent_name=self.role,
            function_name="decide_bid",
            context=ctx,
            expect="bool",
        )

    def respond_to_offer(self, initiator_msg: NegotiationMessage) -> float:
        item    = initiator_msg.negotiation_item
        rnd     = initiator_msg.round
        my_cost = self.specialties.get(item, 0)

        history = self._negotiation_history.get(item, [])
        my_prev = history[-1]["my_response"] if history else None
        num_competitors = self._won_auctions.get(item, {}).get("num_selected", 1)

        ctx = {
            "role":                 self.role,
            "negotiation_item":     item,
            "negotiation_round":    rnd,
            "initiator_offer":      initiator_msg.offer,
            "my_cost":              my_cost,
            "my_previous_response": my_prev,
            "num_competitors":      num_competitors,
            "negotiation_history":  history,
        }
        response = call_llm(
            prompt_name="company_respond_to_offer",
            agent_name=self.role,
            function_name="respond_to_offer",
            context=ctx,
            expect="float",
        )

        self._negotiation_history.setdefault(item, []).append({
            "round":           rnd,
            "initiator_offer": initiator_msg.offer,
            "my_response":     response,
        })
        return response

    # ---- notifications (state-only, no LLM) ----

    def notify_won_auction(self, auction_item: str, auction_round: int,
                           num_selected: int) -> None:
        self._won_auctions[auction_item] = {
            "round":        auction_round,
            "num_selected": num_selected,
        }
        self._log("won auction for %s in round %d (%d companies selected)",
                  auction_item, auction_round, num_selected)

    def notify_contract_assigned(self, construction_item: str, price: float) -> None:
        self._contracts[construction_item] = price
        self._log("contract assigned: %s @ %.2f", construction_item, price)

    def notify_negotiation_lost(self, construction_item: str) -> None:
        self._lost_negotiations.append(construction_item)
        self._log("negotiation lost: %s", construction_item)
