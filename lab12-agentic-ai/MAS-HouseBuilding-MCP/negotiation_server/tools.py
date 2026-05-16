from typing import Dict, Any
from negotiation_server.state import NegotiationState
from shared.types import AgentState, OfferType


class NegotiationTools:
    def __init__(self, state: NegotiationState):
        self.state = state

    def make_offer(
        self,
        from_agent: str,
        to_agent: str,
        price: float,
        offer_type: str,
    ) -> Dict[str, Any]:
        """Submit offer in negotiation."""
        try:
            offer_enum = OfferType[offer_type.upper()]
        except KeyError:
            return {
                "status": "error",
                "message": f"Invalid offer type: {offer_type}",
            }

        if from_agent not in self.state.agent_states:
            return {"status": "error", "message": f"{from_agent} not in negotiation"}

        if price < 0:
            return {"status": "error", "message": "Price must be non-negative"}

        self.state.add_offer(from_agent, to_agent, price, offer_enum)
        self.state.agent_states[from_agent].last_action = (
            f"made {offer_type} offer at {price}"
        )
        self.state.agent_states[from_agent].last_offer_seen = str(price)

        if offer_enum == OfferType.ACCEPT:
            self.state.set_agreement(from_agent, price)
            return {
                "status": "success",
                "message": f"{from_agent} accepted at {price}",
                "agreement_reached": True,
            }

        return {
            "status": "success",
            "message": f"{from_agent} made {offer_type} offer at {price}",
            "agreement_reached": False,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current negotiation status."""
        return {
            "session_id": self.state.session_id,
            "task": self.state.task.name,
            "bidders": self.state.bidders,
            "current_bidder": self.state.current_bidder(),
            "offers": [
                {
                    "from": o.from_agent,
                    "to": o.to_agent,
                    "price": o.price,
                    "type": o.offer_type.value,
                }
                for o in self.state.offers
            ],
            "agreed_price": self.state.agreed_price,
            "agreed_contractor": self.state.agreed_contractor,
            "history": self.state.history,
        }
