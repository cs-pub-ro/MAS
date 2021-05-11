from typing import Callable, List

from commons import CommonsAgent, CommonsPerception
from communication import AgentAction
import math
import random


class StudentAgent(CommonsAgent):
    def __init__(self, agent_id):
        super(StudentAgent, self).__init__(agent_id)

        self.prev_quantity = 0

    def specify_share(self, perception: CommonsPerception) -> float:
        factor = math.log(perception.resource_quantity) / math.log(1e+4)

        share = 0.1 / perception.num_commons * factor / 2

        return share * random.uniform(1, 3)

    def negotiation_response(self, negotiation_round: int, perception: CommonsPerception,
                             utility_func: Callable[[float, float, List[float]], float]) -> AgentAction:
        # TODO: return an AgentAction, whereby the agent can specify what his revised consumption share is, as
        #  well as what he thinks other agents should consume, in the form of a consumption_adjustment dict
        #  Attention: if you specify a consumption_adjustment dict, you have to make sure that it sums up to 0
        #  (i.e. your agent thinks somebody should conusme less and somebody more)

        factor = math.log(perception.resource_quantity) / math.log(1e+4)

        share = 0.1 / perception.num_commons * factor

        lower_share = {}
        higher_share = {}

        for agent_id, agent_share in perception.resource_shares.items():
            if agent_share < share:
                lower_share[agent_id] = agent_share
            elif agent_share > share:
                higher_share[agent_id] = agent_share

        consumption_adjustment = {agent_id + 1: 0 for agent_id in range(perception.num_commons)}

        if self.id in lower_share:
            my_diff = share - lower_share[self.id]

            for high_agent_id, high_agent_share in higher_share.items():
                high_diff = high_agent_share - share

                if my_diff <= high_diff:
                    consumption_adjustment[self.id] += my_diff * perception.resource_quantity
                    consumption_adjustment[high_agent_id] -= my_diff * perception.resource_quantity
                    break
                else:
                    consumption_adjustment[self.id] += high_diff * perception.resource_quantity
                    consumption_adjustment[high_agent_id] -= high_diff * perception.resource_quantity
                    my_diff -= high_diff

        no_action = True if all(s == 0 for s in consumption_adjustment.values()) else False

        return AgentAction(self.id, resource_share=share,
                           consumption_adjustment=consumption_adjustment, no_action=no_action)


