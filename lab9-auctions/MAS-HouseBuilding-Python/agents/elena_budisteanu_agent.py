from typing import Callable, List

from commons import CommonsAgent, CommonsPerception
from communication import AgentAction
import random


class StudentAgent(CommonsAgent):
    def __init__(self, agent_id):
        super(StudentAgent, self).__init__(agent_id)
        self.desired_share = None;

    def specify_share(self, perception: CommonsPerception) -> float:
        ## TODO: return the share that this agent wants to consume at a start of a environment turn

        # The optimum is 1 /(2 * number_of_agents)
        self.desired_share = 1 / (2 * perception.num_commons);
        
        return self.desired_share;

    def negotiation_response(self, negotiation_round: int, perception: CommonsPerception,
                             utility_func: Callable[[float, float, List[float]], float]) -> AgentAction:
        # TODO: return an AgentAction, whereby the agent can specify what his revised consumption share is, as
        #  well as what he thinks other agents should consume, in the form of a consumption_adjustment dict
        #  Attention: if you specify a consumption_adjustment dict, you have to make sure that it sums up to 0
        #  (i.e. your agent thinks somebody should conusme less and somebody more)


        # The desired share of all the other agents
        optimum_share = 1/(2 * perception.num_commons);
        
        # dictionary that holds the adjustments for all agents
        # adjustments[ag_id] = p    => agent ag_id should decrease his share with p
        adjustments = {}

        for ag_id in perception.resource_shares.keys():
            if (ag_id == self.id):
                continue;
            other_agent_share = perception.resource_shares[ag_id];
            adjustments[ag_id] = optimum_share - other_agent_share;
            
        all_zero = True;

        # check if all the adjustments are low enough, so that the agents can stop negociating
        for ag_id in adjustments.keys() :
            if (abs(adjustments[ag_id]) > 0.001):
                all_zero = False;

        # Add own adjustment
        adjustments[self.id] = 0.0;
        if (all_zero):
            # Negociation has stopped, so send No_Action as well.
            return AgentAction(self.id, resource_share=self.desired_share, no_action=True, consumption_adjustment=adjustments)


        return AgentAction(self.id, resource_share=self.desired_share, consumption_adjustment=adjustments);


