from typing import Callable, List

from commons import CommonsAgent, CommonsPerception
from communication import AgentAction

import random


class StudentAgent(CommonsAgent):
    def __init__(self, agent_id):
        super(StudentAgent, self).__init__(agent_id)
        self.best = None

    def specify_share(self, perception: CommonsPerception) -> float:
        ## TODO: return the share that this agent wants to consume at a start of a environment turn

        # We randomly initialize some agents to use a lower share than ideal, in order to test the adjustments
        self.best = 1 / (perception.num_commons + 1)
        if self.id == 1 or self.id == 2:
            self.best = 0.1
        return self.best

    def negotiation_response(self, negotiation_round: int, perception: CommonsPerception,
                             utility_func: Callable[[float, float, List[float]], float]) -> AgentAction:
        # TODO: return an AgentAction, whereby the agent can specify what his revised consumption share is, as
        #  well as what he thinks other agents should consume, in the form of a consumption_adjustment dict
        #  Attention: if you specify a consumption_adjustment dict, you have to make sure that it sums up to 0
        #  (i.e. your agent thinks somebody should conusme less and somebody more)
        suggestions = perception.aggregate_adjustment
        shares = perception.resource_shares

        print("round:", negotiation_round)
        print("utility_func:",
              utility_func(perception.resource_quantity, shares[self.id], shares.values()))

        # We update the shares according to the adjustments (the rich will give to the poor a part of their share)
        total_utility = 0
        utilities = {}
        poor = []
        rich = []

        for crt_id in shares:
            crt_utility = utility_func(perception.resource_quantity,
                                          shares[crt_id],
                                          shares.values())
            utilities[crt_id] = crt_utility
            total_utility += crt_utility
        mean_utility = total_utility / perception.num_commons

        for crt_id in utilities:
            if utilities[crt_id] > mean_utility:
                rich.append(crt_id)
            else:
                poor.append(crt_id)

        if len(poor) % 2 == 1:
            poor.pop()
        if len(rich) % 2 == 1:
            rich.pop()

        if not poor or not rich:
            return AgentAction(self.id,
                               no_action=True)

        suma_poor = 0
        suma_rich = 0
        for sarac_id in poor:
            suma_poor += shares[sarac_id]
        for bogat_id in rich:
            suma_rich += shares[bogat_id]

        diff = suma_rich - suma_poor
        delta_poor = diff / 2 / len(poor)
        delta_rich = diff / 2 / len(rich)

        consumption_adjustment = {}
        for sarac_id  in poor:
            consumption_adjustment[sarac_id] = delta_poor
        for bogat_id in rich:
            consumption_adjustment[bogat_id] = - delta_rich

        # If we have received any suggestions, we will take them into account in a percentage of cases
        probability = 0.5
        if suggestions and random.random() < probability:
            self.best = self.best + suggestions[self.id]

        if self.best <= 0:
            self.best = 0.1
            print("ERROR", self.best, self.id, suggestions)

        return AgentAction(self.id,
                           resource_share=self.best,
                           consumption_adjustment=consumption_adjustment,
                           no_action=False)