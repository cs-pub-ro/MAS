from typing import Callable, List

from commons import CommonsAgent, CommonsPerception
from communication import AgentAction
import random
from functools import reduce 

class StudentAgent(CommonsAgent):
    def __init__(self, agent_id):
        super(StudentAgent, self).__init__(agent_id)
        self.share = 0
        self.round = 0

    def specify_share(self, perception: CommonsPerception) -> float:
        ## TODO: return the share that this agent wants to consume at a start of a environment turn
        num_agents = perception.num_commons
        self.share =  random.uniform(0, 1/num_agents)
        # take into consideration the round number 
        # shared resource and round number are inversely 
        # proportional  
        if self.round > 0:
            self.share /= self.round
        return self.share

    def negotiation_response(self, negotiation_round: int, perception: CommonsPerception,
                             utility_func: Callable[[float, float, List[float]], float]) -> AgentAction:
        # TODO: return an AgentAction, whereby the agent can specify what his revised consumption share is, as
        #  well as what he thinks other agents should consume, in the form of a consumption_adjustment dict
        #  Attention: if you specify a consumption_adjustment dict, you have to make sure that it sums up to 0
        #  (i.e. your agent thinks somebody should conusme less and somebody more)
        resource_quantity = perception.resource_quantity
        resource_shares = perception.resource_shares
        num_agents = perception.num_commons

        self.utility_score = utility_func(resource_quantity, self.share, [*resource_shares.values()])
        if self.utility_score > 50: 
            return AgentAction(self.id, resource_share=0, no_action=True)

        adjustments = perception.aggregate_adjustment

        treshold = 1 / num_agents
        consumption_adjustment = {agent_id: 0 for agent_id in resource_shares}
        total_adjust = reduce(lambda acc, el: (acc[0] + (el - treshold), acc[1] + 1) if (el > treshold) else acc, [*resource_shares.values()], (0, 0))
        to_adjust = total_adjust[0] / (num_agents - total_adjust[1])
        # print(to_adjust)
        if to_adjust != 0:
            for agent_id in resource_shares:
                if resource_shares[agent_id] > treshold:
                    consumption_adjustment[agent_id] = -1 *  (resource_shares[agent_id] - treshold)
                else:
                    consumption_adjustment[agent_id] = to_adjust
            # print(sum(consumption_adjustment.values()))
            self.share += consumption_adjustment[self.id]
            return AgentAction(self.id, resource_share=self.share, consumption_adjustment=consumption_adjustment)
                    

        return AgentAction(self.id, resource_share=0, no_action=True)
        


    def inform_round_finished(self, negotiation_round: int, perception: CommonsPerception):
        self.round += 1
        # print(negotiation_round, self.round)
