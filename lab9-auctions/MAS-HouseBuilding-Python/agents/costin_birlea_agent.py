from typing import Callable, List

from commons import CommonsAgent, CommonsPerception
from communication import AgentAction

import math

def utility(k, n):
    utility = math.log(N)

class StudentAgent(CommonsAgent):
    share = 0

    def __init__(self, agent_id):
        super(StudentAgent, self).__init__(agent_id)

    def specify_share(self, perception: CommonsPerception) -> float:
        ## TODO: return the share that this agent wants to consume at a start of a environment turn

        remaining = perception.resource_remaining
        quantity = perception.resource_quantity
        num_commons = perception.num_commons
        self.share = float(0.1 / num_commons)

        return self.share

    def negotiation_response(self, negotiation_round: int, perception: CommonsPerception,
                             utility_func: Callable[[float, float, List[float]], float]) -> AgentAction:
        # TODO: return an AgentAction, whereby the agent can specify what his revised consumption share is, as
        #  well as what he thinks other agents should consume, in the form of a consumption_adjustment dict
        #  Attention: if you specify a consumption_adjustment dict, you have to make sure that it sums up to 0
        #  (i.e. your agent thinks somebody should conusme less and somebody more)

        remaining = perception.resource_remaining
        quantity = perception.resource_quantity
        num_commons = perception.num_commons
        shares = perception.resource_shares
        aggregate_adjustment = perception.aggregate_adjustment
        round_finished = perception.round_finished


        if round_finished:
            return AgentAction(self.id, resource_share=self.share, no_action=True)

        # We consider that my initial share is the ideal one
        my_share = float(self.share)
        consumption_adjustment = {}
        no_action = True

        diff = float(0)
        total = float(0)
        for (agent_id, share) in shares.items():
            total = float(total + share)
            if (share > my_share * 2):
                new_share = float(share / 2)
                consumption_adjustment[agent_id] = float(new_share)
                diff = float(diff + new_share)
                no_action = False
            else:
                # No Negotiation to those with whom we agree
                consumption_adjustment[agent_id] = float(share)

        new_total = float(0)
        parts = float(diff / num_commons)
        for i in range(1, num_commons + 1):
            consumption_adjustment[i] = float(consumption_adjustment[i] + parts)
            new_total = float(new_total + consumption_adjustment[i])

        print(total, new_total, float(total - new_total))

        if no_action:
            return AgentAction(self.id, resource_share=0, no_action=True)
        else:
            return AgentAction(self.id, resource_share=0, consumption_adjustment = consumption_adjustment, no_action=False)


class BastardAgent(CommonsAgent):
    def __init__(self, agent_id):
        super(BastardAgent, self).__init__(agent_id)

    def specify_share(self, perception: CommonsPerception) -> float:
        ## TODO: return the share that this agent wants to consume at a start of a environment turn

        remaining = perception.resource_remaining
        quantity = perception.resource_quantity
        num_commons = perception.num_commons

        return float(0.5 / num_commons)

    def negotiation_response(self, negotiation_round: int, perception: CommonsPerception,
                             utility_func: Callable[[float, float, List[float]], float]) -> AgentAction:
        # TODO: return an AgentAction, whereby the agent can specify what his revised consumption share is, as
        #  well as what he thinks other agents should consume, in the form of a consumption_adjustment dict
        #  Attention: if you specify a consumption_adjustment dict, you have to make sure that it sums up to 0
        #  (i.e. your agent thinks somebody should conusme less and somebody more)

        remaining = perception.resource_remaining
        quantity = perception.resource_quantity
        num_commons = perception.num_commons
        shares = perception.resource_shares
        aggregate_adjustment = perception.aggregate_adjustment
        round_finished = perception.round_finished

        #print(remaining, quantity, num_commons, shares, aggregate_adjustment, round_finished)
        
        return AgentAction(self.id, resource_share=0, no_action=True)
