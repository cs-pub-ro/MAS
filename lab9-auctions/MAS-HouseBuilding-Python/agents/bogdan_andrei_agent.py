from copy import deepcopy
from typing import Callable, List

from commons import CommonsAgent, CommonsPerception
from communication import AgentAction


import random

class StudentAgent(CommonsAgent):
    def __init__(self, agent_id):
        super(StudentAgent, self).__init__(agent_id)

    def specify_share(self, perception: CommonsPerception) -> float:
        ## TODO: return the share that this agent wants to consume at a start of a environment turn
        # all agents will try to acquire all resources
        value = random.random() / 4
        # print(str(self.id) + ": " + str(value))
        return value

    def negotiation_response(self, negotiation_round: int, perception: CommonsPerception,
                             utility_func: Callable[[float, float, List[float]], float]) -> AgentAction:
        # TODO: return an AgentAction, whereby the agent can specify what his revised consumption share is, as
        #  well as what he thinks other agents should consume, in the form of a consumption_adjustment dict
        #  Attention: if you specify a consumption_adjustment dict, you have to make sure that it sums up to 0
        #  (i.e. your agent thinks somebody should conusme less and somebody more)

        # print("Quantity: " + str(perception.resource_quantity))
        # print("Remaining: " + str(perception.resource_remaining))
        # print("Shares: " + str(perception.resource_shares))
        # print("Aggregate: " + str(perception.aggregate_adjustment))

        all_utilities = deepcopy(perception.resource_shares)
        # print("All utilities: " + str(all_utilities))

        # compute the utility for the current agent
        utility = utility_func(perception.resource_remaining, all_utilities[self.id], all_utilities.values())
        # print("Utility: " + str(utility))
        if utility > 0:
            return AgentAction(self.id, resource_share=all_utilities[self.id], consumption_adjustment={id: 0 for id in all_utilities}, no_action=True)


        # adjust the current agent's share
        steps = 10
        while utility < 0 and steps > 0:
            if utility < 0:
                # require some shares from others
                new_share = 0
                for id in all_utilities:
                    if id != self.id:
                        new_share += all_utilities[id] / len(all_utilities)
                        all_utilities[id] -= all_utilities[id] / len(all_utilities)

                all_utilities[self.id] += new_share
            else:
                # Give some of my share to anyone else that is in need
                share = all_utilities[self.id] / len(all_utilities)
                all_utilities[self.id] -= share

                share /= len(all_utilities) - 1
                for id in all_utilities:
                    if id != self.id:
                        all_utilities[id] += share

            # print("All utilities: " + str(all_utilities))

            utility = utility_func(perception.resource_remaining, all_utilities[self.id], all_utilities.values())
            # print("Utility: " + str(utility))

            steps -= 1

        if utility > 0:
            adjustments = {id: all_utilities[id] - perception.resource_shares[id] for id in all_utilities}
        else:
            adjustments = deepcopy(perception.resource_shares)

            share = adjustments[self.id] / (len(adjustments) - 1)
            adjustments[self.id] = - adjustments[self.id]

            for id in adjustments:
                if id != self.id:
                    adjustments[id] = share

        # print(str(self.id) + " -> Adjustments: " + str(adjustments))
        # print()
        # print()

        return AgentAction(self.id, resource_share=adjustments[self.id], consumption_adjustment=adjustments,
                           no_action=False)




    def inform_round_finished(self, negotiation_round: int, perception: CommonsPerception):
        pass
