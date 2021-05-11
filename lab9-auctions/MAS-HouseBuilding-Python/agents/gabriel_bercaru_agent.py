from typing import Callable, List

from commons import CommonsAgent, CommonsPerception
from communication import AgentAction

from random import random

import numpy as np

class StudentAgent_GabrielBercaru(CommonsAgent):
    def __init__(self, agent_id):
        super(StudentAgent_GabrielBercaru, self).__init__(agent_id)
        self.prev_share = 0.0

    def specify_share(self, perception: CommonsPerception) -> float:
        ## TODO: return the share that this agent wants to consume at a start of a environment turn
        my_share = 0.1 #random() / 5
        vi = np.log(my_share) + np.log(perception.resource_remaining * (1 - my_share))
        prev_vi = 0
        eps = 0.001
        min_share = 0.01
        alpha = 0.01

        while np.abs(vi - prev_vi > eps) and my_share > min_share:
            prev_vi = vi
            dvi_dki = 1 / np.log(my_share) + 1 / np.log(perception.resource_remaining - (1 * my_share))
            #print("dvi_dki = {}".format(dvi_dki))
            my_share += dvi_dki * alpha
            vi = np.log(my_share) + np.log(perception.resource_remaining * (1 - my_share))
            #print("Diff: {}".format(np.abs(vi - prev_vi)))

        self.prev_share = my_share

        return random() / 8
        return my_share

    def negotiation_response(self, negotiation_round: int, perception: CommonsPerception,
                             utility_func: Callable[[float, float, List[float]], float]) -> AgentAction:
        # TODO: return an AgentAction, whereby the agent can specify what his revised consumption share is, as
        #  well as what he thinks other agents should consume, in the form of a consumption_adjustment dict
        #  Attention: if you specify a consumption_adjustment dict, you have to make sure that it sums up to 0
        #  (i.e. your agent thinks somebody should conusme less and somebody more)

        alpha = 0.0001
        eps = 0.001
        min_grad = -10.0
        max_grad = 10.0
        min_share = 0.02

        consumption_adjustment = {}
        other_shares = perception.resource_shares
        agg_adjst = perception.aggregate_adjustment
        my_share = other_shares[self.id]

        if agg_adjst:
            for k, v in other_shares.items():
                other_shares[k] = other_shares[k] + agg_adjst[k]

        #vi = utility_func(perception.resource_remaining, my_share, list(other_shares.values()))
        vi = np.log(my_share) + np.log(perception.resource_remaining * (1 - (my_share + sum(other_shares.values()))))
        w = np.log(my_share) + sum(list(map(lambda x : np.log(x), list(other_shares.values())))) \
            + (1 + len(other_shares)) * np.log(perception.resource_remaining - (my_share + sum(list(map(lambda x : np.log(x), list(other_shares.values()))))))

        all_shares = list(other_shares.values())
        all_grads = list(zip(list(perception.resource_shares.keys()), list(map(lambda x : alpha / np.log(x) + alpha * len(all_shares) * perception.resource_remaining * x / np.log(perception.resource_remaining * (1 - sum(all_shares))), all_shares))))

        #print("All_grads: " + str(all_grads))

        all_grads = list(map(lambda x : (x[0], x[1]) if x[1] > min_grad and x[1] < max_grad else (x[0], 0.0), all_grads))

        num_neg = len(list(filter(lambda x : x[1] < 0.0, all_grads)))
        num_pos = len(list(filter(lambda x : x[1] >= 0.0, all_grads)))

        if num_neg == 0 or num_pos == 0:
            all_grads = list(map(lambda x : (x[0], x[1] - (max(all_grads, key = lambda item : (item[1], item[0]))[1] - min(all_grads, key = lambda item : (item[1], item[0]))[1]) / 2), all_grads))

        pos_grad = list(filter(lambda x : x[1] >= 0.0, all_grads))
        neg_grad = list(filter(lambda x : x[1] < 0.0, all_grads))
        sum_pos = sum(list(map(lambda x : x[1], pos_grad)))
        sum_neg = sum(list(map(lambda x : x[1], neg_grad)))

        #print("All_grads2: " + str(all_grads))

        if sum_pos > abs(sum_neg):
            ratio = abs(sum_neg) / sum_pos
            pos_grad = list(map(lambda x : (x[0], x[1] * ratio), pos_grad))
        else:
            ratio = sum_pos / abs(sum_neg)
            neg_grad = list(map(lambda x : (x[0], x[1] * ratio), neg_grad))

        for x in pos_grad:
            consumption_adjustment[x[0]] = x[1]
        for x in neg_grad:
            consumption_adjustment[x[0]] = x[1]

        
        for k, v in consumption_adjustment.items():
            if other_shares[k] + v < min_share:
                consumption_adjustment[k] = (min_share - (other_shares[k] + v))#-(other_shares[k] - min_share)

                penalty_for_others = consumption_adjustment[k] / (len(consumption_adjustment) - 1)#(min_share - (other_shares[k] + v)) / (len(consumption_adjustment) - 1)
                for k2, v2 in consumption_adjustment.items():
                    if k2 != k:
                        consumption_adjustment[k2] = consumption_adjustment[k2] - penalty_for_others


        if all(list(map(lambda x : abs(x) < eps, list(consumption_adjustment.values())))):
            return AgentAction(self.id, resource_share=0, no_action=True)

        return AgentAction(self.id, resource_share=other_shares[self.id] + consumption_adjustment[self.id], consumption_adjustment=consumption_adjustment, no_action=False)


