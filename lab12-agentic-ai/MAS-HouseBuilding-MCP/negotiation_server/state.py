from dataclasses import dataclass, field
from typing import Dict, List, Any
from shared.types import Task, Offer, OfferType, AgentState, Phase


@dataclass
class NegotiationState:
    session_id: str
    task: Task
    bidders: List[str] = field(default_factory=list)
    current_bidder_index: int = 0
    offers: List[Offer] = field(default_factory=list)
    agreed_price: float = 0.0
    agreed_contractor: str = ""
    agent_states: Dict[str, AgentState] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    phase: Phase = Phase.NEGOTIATION

    def add_offer(
        self,
        from_agent: str,
        to_agent: str,
        price: float,
        offer_type: OfferType,
    ):
        offer = Offer(
            from_agent=from_agent,
            to_agent=to_agent,
            price=price,
            offer_type=offer_type,
        )
        self.offers.append(offer)
        self.history.append(
            {
                "from": from_agent,
                "to": to_agent,
                "price": price,
                "type": offer_type.value,
                "event": "offer_made",
            }
        )

    def set_agreement(self, contractor: str, price: float):
        self.agreed_contractor = contractor
        self.agreed_price = price
        self.history.append(
            {
                "contractor": contractor,
                "price": price,
                "event": "agreement_reached",
            }
        )

    def next_bidder(self):
        self.current_bidder_index += 1

    def current_bidder(self) -> str:
        if self.current_bidder_index < len(self.bidders):
            return self.bidders[self.current_bidder_index]
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task": {"name": self.task.name, "budget": self.task.budget},
            "bidders": self.bidders,
            "current_bidder": self.current_bidder(),
            "offers": [
                {
                    "from": o.from_agent,
                    "to": o.to_agent,
                    "price": o.price,
                    "type": o.offer_type.value,
                }
                for o in self.offers
            ],
            "agreed_price": self.agreed_price,
            "agreed_contractor": self.agreed_contractor,
            "history": self.history,
            "phase": self.phase.value,
        }
