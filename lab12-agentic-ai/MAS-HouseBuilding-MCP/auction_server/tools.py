from typing import Dict, Any
from auction_server.state import AuctionState


class AuctionTools:
    def __init__(self, state: AuctionState):
        self.state = state

    def propose_budget(self, price: float) -> Dict[str, Any]:
        """ACME proposes a budget price for this round."""
        if price <= self.state.current_price:
            return {
                "status": "error",
                "message": f"Price must be > {self.state.current_price}",
            }

        if price > self.state.task.budget:
            return {
                "status": "error",
                "message": f"Price must be <= budget {self.state.task.budget}",
            }

        if self.state.current_round >= self.state.max_rounds:
            return {
                "status": "error",
                "message": "Maximum rounds reached",
            }

        self.state.start_new_round(price)
        return {
            "status": "success",
            "message": f"Budget proposed at {price}",
            "round": self.state.current_round,
            "price": price,
        }

    def bid(self, company: str) -> Dict[str, Any]:
        """Company submits a bid at the current proposed price."""
        if company not in self.state.agent_states:
            return {"status": "error", "message": f"{company} not registered"}

        if self.state.current_round == 0:
            return {"status": "error", "message": "No active auction round"}

        self.state.add_bid(company, self.state.current_price)
        self.state.agent_states[company].last_action = (
            f"bid at {self.state.current_price}"
        )

        return {
            "status": "success",
            "message": f"{company} bid {self.state.current_price}",
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current auction status."""
        bids = [
            {"company": b.company, "price": b.price} for b in self.state.bids
        ]

        return {
            "session_id": self.state.session_id,
            "phase": self.state.phase.value,
            "task": self.state.task.name,
            "round": self.state.current_round,
            "current_price": self.state.current_price,
            "bids": bids,
            "max_rounds": self.state.max_rounds,
            "winner": self.state.winner,
            "history": self.state.history,
        }
