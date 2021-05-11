from typing import List, Dict, Any

from scipy.stats._multivariate import special_ortho_group_frozen

from agents import HouseOwnerAgent, CompanyAgent
from communication import NegotiationMessage


class MyACMEAgent(HouseOwnerAgent):

    def __init__(self, role: str, budget_list: List[Dict[str, Any]]):
        super(MyACMEAgent, self).__init__(role, budget_list)
        self.auction_result = {}
        self.auction_result_budget = {}

    def propose_item_budget(self, auction_item: str, auction_round: int) -> float:
        
        initial_item_budget = self.budget_dict[auction_item]
        
        # Dutch descending auction
        if auction_round == 0 :
            self.auction_result_budget[auction_item] = 0.3*initial_item_budget
            return 0.3*initial_item_budget
        elif auction_round == 1 : 
            self.auction_result_budget[auction_item] = 0.6*initial_item_budget
            return 0.6*initial_item_budget
        else:
            self.auction_result_budget[auction_item] = 1.*initial_item_budget
            return 1.*initial_item_budget      

    def notify_auction_round_result(self, auction_item: str, auction_round: int, responding_agents: List[str]):
        self.auction_result[auction_item] = responding_agents

    def provide_negotiation_offer(self, negotiation_item: str, partner_agent: str, negotiation_round: int) -> float:
        
        # Variant Dutch auction means start with a lower price and increase round by round
        
        if negotiation_round == 0:
            return 0.3*self.auction_result_budget[negotiation_item]
        elif negotiation_round==1 :
            return 0.6*self.auction_result_budget[negotiation_item]
        else:
            return 1.0*self.auction_result_budget[negotiation_item]

    def notify_negotiation_winner(self, negotiation_item: str, winning_agent: str, winning_offer: float) -> None:
        pass


class MyCompanyAgent(CompanyAgent):

    def __init__(self, role: str, specialties: List[Dict[str, Any]]):
        super(MyCompanyAgent, self).__init__(role, specialties)
        self.won_contract = {} # history of the won contract's
        self.auction_result = {}
        
    def decide_bid(self, auction_item: str, auction_round: int, item_budget: float) -> bool:
        
        company_budget = self.specialties[auction_item]
        
        # if the company have been selected for at least one auction
        if len(self.auction_result) >= 1:
            
            # if the asked price is too high for the company refuse the offer
            if item_budget > company_budget :
                return False
            
            # otherwise accept it
            else:
                return True          
            
        # if the company, don't have any contract  
        else:
            return True

    def notify_won_auction(self, auction_item: str, auction_round: int, num_selected: int):
        self.auction_result[auction_item] = num_selected

    def respond_to_offer(self, initiator_msg: NegotiationMessage) -> float:
        negotiation_item = initiator_msg.negotiation_item
        nr_round = initiator_msg.round
        offer = initiator_msg.offer
        
        if nr_round==0 :
            # Try to negotiate
            return offer*1.30
        elif nr_round == 1:
            # Moderate your offer
            return offer*1.2
        else:
            # if you don't have any contract, accept anything
            if len(won_contract[negotiation_item])==0:
                return offer
            
            # if you have one contract 
            elif len(won_contract[negotiation_item])==1:
                return offer*1.05
            
            # if you have more than two contract
            else:
                return offer*1.1

    def notify_contract_assigned(self, construction_item: str, price: float) -> None:
        self.won_contract[construction_item] = price

    def notify_negotiation_lost(self, construction_item: str) -> None:
        pass
