from typing import Callable, List

from commons import CommonsAgent, CommonsPerception
from communication import AgentAction


class StudentAgent(CommonsAgent):
    def __init__(self, agent_id):
        super(StudentAgent, self).__init__(agent_id)
        self.initial_resource_quantity = None
        self.previous_agg_adjustment = None
        self.last_share = None
        self.last_utility = None

    def specify_share(self, perception: CommonsPerception) -> float:
        if self.initial_resource_quantity is None:
            self.initial_resource_quantity = perception.resource_quantity

        ## TODO: return the share that this agent wants to consume at a start of a environment turn
        if perception.resource_quantity >= self.initial_resource_quantity:
            share = 1 / (2 * perception.num_commons)
        else:
            share = 0.001

        self.last_share = share

        return share

    def negotiation_response(self, negotiation_round: int, perception: CommonsPerception,
                             utility_func: Callable[[float, float, List[float]], float]) -> AgentAction:
        # TODO: return an AgentAction, whereby the agent can specify what his revised consumption share is, as
        #  well as what he thinks other agents should consume, in the form of a consumption_adjustment dict
        #  Attention: if you specify a consumption_adjustment dict, you have to make sure that it sums up to 0
        #  (i.e. your agent thinks somebody should conusme less and somebody more)
        if negotiation_round == 0:
            self.previous_agg_adjustment = None
        
        if self.last_utility is None:
            self.last_utility = utility_func(perception.resource_quantity, self.last_share, perception.resource_shares)

        if perception.resource_quantity >= self.initial_resource_quantity:
            my_share = 1 / (2 * perception.num_commons)
        else:
            my_share = 0.001

        adjustments = {}
        if perception.aggregate_adjustment:
            adjustment_for_me = perception.aggregate_adjustment[self.id]
            if adjustment_for_me is not None:
                if adjustment_for_me > 0:
                    adjustments[self.id] = adjustment_for_me

            for agent, share in perception.resource_shares.items():
                adjustment = perception.aggregate_adjustment[agent]
                if adjustment is None:
                    continue
                if share == my_share:
                    continue
                if share > my_share:
                    if adjustment < 0:
                        adjustments[agent] = adjustment
                    elif adjustment > 0:
                        adjustments[agent] = (share - my_share) * perception.num_commons

        else:
            total_amount = 0
            for agent, share in perception.resource_shares.items():
                if share == my_share:
                    continue
                if share > my_share:
                    amount = (my_share - share) * perception.num_commons
                    adjustments[agent] = amount
                    total_amount += amount
            my_share -= total_amount
            adjustments[self.id] = my_share

        return AgentAction(self.id, resource_share=my_share, consumption_adjustment = adjustments, no_action=True)


