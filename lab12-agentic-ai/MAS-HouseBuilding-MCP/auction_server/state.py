from dataclasses import dataclass, field
from typing import Dict, List, Any
from shared.types import Task, Bid, AgentState, Phase


@dataclass
class AuctionState:
    session_id: str
    task: Task
    current_round: int = 0
    max_rounds: int = 3
    current_price: float = 0.0
    bids: List[Bid] = field(default_factory=list)
    winner: str = ""
    agent_states: Dict[str, AgentState] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    phase: Phase = Phase.AUCTION

    def start_new_round(self, acme_price: float):
        self.current_round += 1
        self.current_price = acme_price
        self.bids = []
        self.history.append(
            {
                "round": self.current_round,
                "price": acme_price,
                "event": "round_started",
            }
        )

    def add_bid(self, company: str, price: float):
        bid = Bid(company=company, price=price)
        self.bids.append(bid)
        self.history.append(
            {
                "round": self.current_round,
                "bidder": company,
                "price": price,
                "event": "bid_submitted",
            }
        )

    def set_winner(self, company: str):
        self.winner = company
        self.history.append(
            {
                "round": self.current_round,
                "winner": company,
                "event": "auction_completed",
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task": {"name": self.task.name, "budget": self.task.budget},
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "current_price": self.current_price,
            "bids": [{"company": b.company, "price": b.price} for b in self.bids],
            "winner": self.winner,
            "history": self.history,
            "phase": self.phase.value,
        }
