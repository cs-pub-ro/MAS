import math
from typing import Callable, List

from commons import CommonsAgent, CommonsPerception
from communication import AgentAction


class StudentAgent(CommonsAgent):
    CONSUMPTION_PERCENTAGE = 0.5

    def __init__(self, agent_id):
        super(StudentAgent, self).__init__(agent_id)
        self.name = "EQUILIBRIUM" + "_" + str(agent_id)
        self.negotiation_rounds = -1
        self.negotiation_history = []

    def specify_share(self, perception: CommonsPerception) -> float:
        """
        Expects the consumption of all agents to be a percentage of all the resources such that future generations will
        still have available resources.
        :param perception: The perception of the world
        :return: Return the share that this agent wants to consume at a start of a environment turn
        """
        return 1.0 / perception.num_commons * StudentAgent.CONSUMPTION_PERCENTAGE

    def negotiation_response(self, negotiation_round: int, perception: CommonsPerception,
                             utility_func: Callable[[float, float, List[float]], float]) -> AgentAction:
        # TODO: return an AgentAction, whereby the agent can specify what his revised consumption share is, as
        #  well as what he thinks other agents should consume, in the form of a consumption_adjustment dict
        #  Attention: if you specify a consumption_adjustment dict, you have to make sure that it sums up to 0
        #  (i.e. your agent thinks somebody should conusme less and somebody more)
        resource_shares = perception.resource_shares.copy()
        adjustments = {}
        actual_percentage = perception.resource_remaining / perception.resource_quantity
        no_action = False

        if perception.aggregate_adjustment:
            for agent_id in resource_shares:
                if resource_shares[agent_id]:
                    resource_shares[agent_id] += perception.aggregate_adjustment[agent_id]

        target_share = 1.0 / perception.num_commons * actual_percentage

        for agent_id in resource_shares:
            agent_share = resource_shares[agent_id]

            if agent_share != target_share:
                adjustments[agent_id] = target_share - agent_share

        if len(adjustments) == 0 or \
                (perception.aggregate_adjustment and
                 len(perception.aggregate_adjustment.items() - self.negotiation_history[-1].items()) >
                 math.ceil(0.25 * perception.num_commons) and
                 negotiation_round < self.negotiation_rounds - 1):
            no_action = True

        self.negotiation_history.append(perception.aggregate_adjustment)

        return AgentAction(self.id, resource_share=0, no_action=no_action, consumption_adjustment=adjustments)

    def inform_round_finished(self, negotiation_round: int, perception: CommonsPerception):
        if perception.aggregate_adjustment:
            self.negotiation_rounds = negotiation_round
        self.negotiation_history = []
