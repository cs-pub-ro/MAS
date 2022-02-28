from typing import List, Dict, Any

from scipy.stats._multivariate import special_ortho_group_frozen

from agents import HouseOwnerAgent, CompanyAgent
from communication import NegotiationMessage


class MyACMEAgent(HouseOwnerAgent):

    def __init__(self, role: str, budget_list: List[Dict[str, Any]]):
        super(MyACMEAgent, self).__init__(role, budget_list)

    def propose_item_budget(self, auction_item: str, auction_round: int) -> float:
        pass

    def notify_auction_round_result(self, auction_item: str, auction_round: int, responding_agents: List[str]):
        pass

    def provide_negotiation_offer(self, negotiation_item: str, partner_agent: str, negotiation_round: int) -> float:
        pass

    def notify_negotiation_winner(self, negotiation_item: str, winning_agent: str, winning_offer: float) -> None:
        pass


class MyCompanyAgent(CompanyAgent):

    def __init__(self, role: str, specialties: List[Dict[str, Any]]):
        super(MyCompanyAgent, self).__init__(role, specialties)

    def decide_bid(self, auction_item: str, auction_round: int, item_budget: float) -> bool:
        pass

    def notify_won_auction(self, auction_item: str, auction_round: int, num_selected: int):
        pass

    def respond_to_offer(self, initiator_msg: NegotiationMessage) -> float:
        pass

    def notify_contract_assigned(self, construction_item: str, price: float) -> None:
        pass

    def notify_negotiation_lost(self, construction_item: str) -> None:
        pass
