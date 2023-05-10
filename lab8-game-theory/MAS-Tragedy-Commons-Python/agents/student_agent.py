from typing import Callable, List

from commons import CommonsAgent, CommonsPerception
from communication import AgentAction


class StudentAgent(CommonsAgent):
    def __init__(self, agent_id):
        super(StudentAgent, self).__init__(agent_id)

    def specify_share(self, perception: CommonsPerception) -> float:
        ## TODO: return the share that this agent wants to consume at a start of a environment turn
        return 0

    def negotiation_response(self, negotiation_round: int, perception: CommonsPerception,
                             utility_func: Callable[[float, float, List[float]], float]) -> AgentAction:
        # TODO: return an AgentAction, whereby the agent can specify what his revised consumption share is, as
        # well as what he thinks other agents should consume, in the form of a consumption_adjustment dict
        #
        # Attention: you must pay attention to the fact the the consumption_adjustment dict may indicate altering
        # the agent_shares in such a way that their sum is greater than 1 or smaller than 0. 
        # You must avoid this, as it will lead to the consumption round being aborted and all agents receiving a 0 utility.
        return AgentAction(self.id, resource_share=0, no_action=True)

    def inform_round_finished(self, negotiation_round: int, perception: CommonsPerception):
        ## information sent to the agent once the current round (including all adjustment rounds) is finished
        pass


