from dataclasses import dataclass, field
from typing import Any, Dict, List
from enum import Enum


class Phase(Enum):
    AUCTION = "auction"
    NEGOTIATION = "negotiation"


class OfferType(Enum):
    OFFER = "offer"
    COUNTER = "counter"
    ACCEPT = "accept"


@dataclass
class Task:
    name: str
    budget: float


@dataclass
class Specialty:
    specialty: str
    cost: float


@dataclass
class Company:
    name: str
    specialties: List[Specialty] = field(default_factory=list)


@dataclass
class Bid:
    company: str
    price: float


@dataclass
class Offer:
    from_agent: str
    to_agent: str
    price: float
    offer_type: OfferType


@dataclass
class AgentState:
    name: str
    joined: bool = False
    active: bool = False
    last_action: str = ""
    last_offer_seen: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joined": self.joined,
            "active": self.active,
            "last_action": self.last_action,
            "last_offer_seen": self.last_offer_seen,
        }
