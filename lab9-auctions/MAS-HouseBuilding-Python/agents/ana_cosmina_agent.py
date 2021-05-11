from copy import deepcopy
from random import random
from typing import Callable, List

from commons import CommonsAgent, CommonsPerception
from communication import AgentAction


class StudentAgent(CommonsAgent):
    def __init__(self, agent_id):
        super(StudentAgent, self).__init__(agent_id)

    """
        Return the share that this agent wants to consume at a start of a environment turn.
        Try to equally divide the resources between the agents and the environment in half of
        the cases, and leave a bit more for the environment in the other half of the cases.
    """

    def specify_share(self, perception: CommonsPerception) -> float:
        if random() < 0.5:
            return 1 / (perception.num_commons + 1)
        return 1 / (2 * perception.num_commons)

    """
        Return an AgentAction, whereby the agent can specify what his revised consumption
        share is, as well as what he thinks other agents should consume, in the form of a
        consumption_adjustment dict.
        Attention: if you specify a consumption_adjustment dict, you have to make sure that
        it sums up to 0 (i.e. your agent thinks somebody should consume less and somebody more).
    """

    def negotiation_response(self, negotiation_round: int, perception: CommonsPerception,
                             utility_func: Callable[[float, float, List[float]], float],
                             verbose=True) -> AgentAction:
        # Based on the perceptions:
        suggestions = deepcopy(perception.aggregate_adjustment)
        shares = deepcopy(perception.resource_shares)

        resource_share = shares[self.id]
        consumption_adjustment = dict.fromkeys(range(1, perception.num_commons + 1), 0)

        #   a) Either revise your own share (based on the received suggestions).
        #      Only listen to suggestions in a third of the cases.
        if suggestions and sum(suggestions.values()) != 0 and suggestions[self.id] and random() < 0.3:
            resource_share = shares[self.id] + suggestions[self.id]
            if verbose:
                print("Agent {0}'s share was {1} and now wants {2}".format(
                    self.id, shares[self.id], resource_share))

        if random() < 0.5:
            return AgentAction(self.id, resource_share=resource_share,
                               consumption_adjustment=consumption_adjustment,
                               no_action=True)

        #   b) And/or communicate changes to the other agents.
        #      Try to balance the utility of all agents in a Robin Hood-like method:
        #      take from the rich ones (i.e. having an utility greater than the average
        #      utility) and divide between the poor ones (i.e. -"- smaller -"-).
        #      When the utility of all agents is balanced, the sum of logs is going to
        #      be greater. Thus, the common utility will also be greater.
        #      Only make adjustments in a third of the cases.

        # Compute the mean utility for an agent.
        sum_utilities = 0
        utilities = {}
        for ag_id in shares:
            ut = utility_func(perception.resource_quantity, shares[ag_id], shares.values())
            utilities[ag_id] = ut
            sum_utilities += ut
        mean_utility = sum_utilities / perception.num_commons

        # Divide the agents into "poor" and "rich" ones, with respect to the mean utility.
        poor_agents = []
        poor_utilities = []
        rich_agents = []
        rich_utilities = []
        for ag_id in utilities:
            ut = utilities[ag_id]
            if ut > mean_utility:
                rich_agents.append(ag_id)
                rich_utilities.append(ut)
            else:
                poor_agents.append(ag_id)
                poor_utilities.append(ut)

        # Because of floating point problems, the division between an odd number of agents
        # will lead to a total utility sum different from one. To overcome this problem, we
        # make sure that we always divide between an even number of agents.
        # In case of oddity, we remove the agent that has the utility closest to the mean
        # (i.e. the richest poor agent and the poorest rich agent).
        if len(rich_agents) % 2 == 1:
            poorest_id = rich_utilities.index(min(rich_utilities))
            rich_agents.pop(poorest_id)
        if len(poor_agents) % 2 == 1:
            richest_id = poor_utilities.index(max(poor_utilities))
            poor_agents.pop(richest_id)

        if not rich_agents or not poor_agents:
            return AgentAction(self.id, resource_share=resource_share,
                               consumption_adjustment=consumption_adjustment, no_action=True)

        from_rich = sum(shares[ag_id] for ag_id in rich_agents)
        to_poor = sum(shares[ag_id] for ag_id in poor_agents)
        delta = from_rich - to_poor
        delta_rich = delta / (2 * len(rich_agents))
        delta_poor = delta / (2 * len(poor_agents))

        for ag_id in range(1, perception.num_commons + 1):
            if ag_id in rich_agents:
                consumption_adjustment[ag_id] -= delta_rich
            elif ag_id in poor_agents:
                consumption_adjustment[ag_id] = delta_poor
        no_action = False

        if random() < 0.5:
            return AgentAction(self.id, resource_share=resource_share,
                               consumption_adjustment=consumption_adjustment,
                               no_action=False)

        return AgentAction(self.id, resource_share=resource_share,
                           consumption_adjustment=consumption_adjustment, no_action=True)














