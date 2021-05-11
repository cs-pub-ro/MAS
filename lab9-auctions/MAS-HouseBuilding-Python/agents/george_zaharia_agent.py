from typing import Callable, List

from commons import CommonsAgent, CommonsPerception
from communication import AgentAction

import random

class StudentAgent(CommonsAgent):
    def __init__(self, agent_id):
        super(StudentAgent, self).__init__(agent_id)

    def specify_share(self, perception: CommonsPerception) -> float:
        ## TODO: return the share that this agent wants to consume at a start of a environment turn

        # determine symmetric Nash equilibrium value
        K = perception.resource_remaining
        n = perception.num_commons
        k_sne = K/(n+1)
        percentage = k_sne/K

        #percentage = random.uniform(0.1, 0.3)

        return percentage

    def negotiation_response(self, negotiation_round: int, perception: CommonsPerception,
                             utility_func: Callable[[float, float, List[float]], float]) -> AgentAction:
        # TODO: return an AgentAction, whereby the agent can specify what his revised consumption share is, as
        #  well as what he thinks other agents should consume, in the form of a consumption_adjustment dict
        #  Attention: if you specify a consumption_adjustment dict, you have to make sure that it sums up to 0
        #  (i.e. your agent thinks somebody should conusme less and somebody more)

        #compute the Pareto optimal consumption of resource
        K = perception.resource_remaining
        if K == 0:
            return AgentAction(self.id, resource_share=perception.resource_shares[self.id], no_action=True)


        n = perception.num_commons
        k_po = K/(2*n)
        optimal_new_resource_share = k_po/K

        # compute the differences between the agent's share and the optimal share
        differences = {}
        for agent in perception.resource_shares.keys():
            if agent != self.id:
                differences[agent] = abs(perception.resource_shares[agent] - optimal_new_resource_share)
            else:
                differences[agent] = abs(perception.resource_shares[agent] - optimal_new_resource_share)/3

        differences = {k: v for k, v in sorted(differences.items(), key=lambda item: item[1], reverse=True)}

        # get the difference average
        diff_average = sum(differences[k] for k in differences.keys())/len(differences.keys())



        checksum = 0
        for elem in differences:
            if differences[elem] < 0.05:
                checksum += 1

        # if every difference is lower than a threshold, the agent takes no action
        if checksum == perception.num_commons or perception.round_finished:# or utility_func(K, optimal_new_resource_share, perception.resource_shares) > 1:
            return AgentAction(self.id, resource_share=optimal_new_resource_share, no_action=True)

        overall_balance = 0

        adjustments = {}

        # compute the adjustment: relatively small adjustments,
        # directly proportional with the aget difference compared to the ideal share
        i = 0
        last_key = None
        for k in differences.keys():
            if i == len(differences.keys()) - 1:
                last_key = k
                break
            agent_change = diff_average * differences[k] * 10
            if perception.resource_shares[k] > optimal_new_resource_share:
                adjustments[k] = -agent_change
                overall_balance -= agent_change
            else:
                adjustments[k] = agent_change
                overall_balance += agent_change
            i += 1

        # the agent is a little greedy, it intends to get a bigger share (but not by much)
        adjustments[last_key] = -overall_balance



        return AgentAction(self.id, resource_share=optimal_new_resource_share, consumption_adjustment=adjustments, no_action=False)


