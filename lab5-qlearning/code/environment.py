from blocksworld import *
from base import Environment, Agent, Perception
import random

import logging.config
import yaml

with open('logging_conf.yaml', 'r') as f:
    log_cfg = yaml.safe_load(f.read())

logging.config.dictConfig(log_cfg)
logger = logging.getLogger("environment")


""" ======================================== Blocksworld agent ======================================== """
class BlocksWorldPerception(Perception):

    # def __init__(self, stack: BlockStack, current_station: Station, previous_action_succeeded: bool):
    #     super(BlocksWorldPerception, self).__init__()
    #
    #     self.visible_stack = stack
    #     self.current_station = current_station
    #     self.previous_action_succeeded = previous_action_succeeded

    def __init__(self, stacks: List[BlockStack], current_station: Station, previous_action_succeeded: bool):
        super(BlocksWorldPerception, self).__init__()

        self.visible_stacks = stacks
        self.current_station = current_station
        self.previous_action_succeeded = previous_action_succeeded


class BlocksWorldAgent(Agent):
    """
    Base class to be implemented by agent implementations. A reactive agent is only defined by its Agent @ to
    perceptions.
    """

    def __init__(self, name = None):
        super(BlocksWorldAgent, self).__init__(name)


    def response(self, perception: BlocksWorldPerception) -> BlocksWorldAction:
        """
        Supplies the agent with perceptions and demands one action from the agent. The environment also specifies if the
        previous action of the agent has succeeded or not.

        :param perception: the perceptions offered by the environment to the agent.
        :return: he agent output, containing the action to perform. Action should be of type
        {@link blocksworld.BlocksWorldAction.Type#NONE} if the agent is not performing an action now,
        but may perform more actions in the future.
        Action should be of type {@link blocksworld.BlocksWorldAction.Type#AGENT_COMPLETED} if the agent will not
        perform any more actions ever in the future.
        """
        raise NotImplementedError("Missing a response")


    def status_string(self):
        """
        :return: a string that is printed at every cycle to show the status of the agent.
        """
        return NotImplementedError("Missing a status string")


    def __str__(self):
        """
        :return: The agent name
        """
        return "A"



class AgentData(object):
    """
    Contains data for each agent in the environment
    """
    def __init__(self, linked_agent: BlocksWorldAgent or None, target: BlocksWorld, initial_station: Station):
        """
        Default constructor.
        :param linked_agent: the agent
        :param target: the desired state of the agent
        :param initial_station: the initial position of the agent (the station at which it is located)
        """
        self.agent = linked_agent
        self.target_state = target
        self.station = initial_station

        self.previous_action_succeeded = True
        self.holding = None


    def __str__(self):
        return "Agent %s at %s holding %s; previous action: %s" \
               % (str(self.agent),
                  str(self.station),
                  str(self.holding) if self.holding else "none",
                  "successful" if self.previous_action_succeeded else "failed")


    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.agent == other.agent
        else:
            return False


class DynamicBlocksWorldEnvironment(object):

    def __init__(self, start_world: BlocksWorld, target_world: BlocksWorld, dynamics: float = 0.5):
        self.init_state = start_world.clone()
        self.world_state = start_world.clone()
        self.target_state = target_world.clone()
        
        self.dynamics = dynamics

        idx = ord("0")
        self.station = Station(str(chr(idx)))

        self.adata = AgentData(linked_agent=None, target=target_world, initial_station=self.station)

    def reset(self):
        self.world_state = self.init_state.clone()
        self.adata.holding = None

    def _get_agent_data(self):
        return self.adata

    def step(self, act: BlocksWorldAction) -> Tuple[frozenset, int, bool]:
        #print("\n".join([str(adata) for adata in self.agents_data]))

        world_stacks = self.world_state.get_stacks()
        reward = REWARD_INVALID
        completed = False

        logger.info("Agent opts for %s" % (str(act)))

        # set previous action succeeded as False, initially
        self.adata.previous_action_succeeded = False

        if (act.get_type() == "putdown" or act.get_type() == "stack") and \
            (not self.adata.holding or self.adata.holding != act.get_first_arg()):
            logger.error("Can't work with that block [%s]; agent is holding [%s]"
                               % (str(act.get_first_arg()), str(self.adata.holding)))
            return self.get_world_state(), REWARD_INVALID, False

        if (act.get_type() == "pickup" or act.get_type() == "unstack") and self.adata.holding:
            logger.error("Can't work with that block [%s]; agent is holding [%s]"
                               % (str(act.get_first_arg()), str(self.adata.holding)))
            return self.get_world_state(), REWARD_INVALID, False

        if act.get_type() == "pickup":
            # modify world; remove station; switch agent to other station.
            block_in_stacks = False
            for stack in world_stacks:
                if act.get_argument() in stack:
                    block_in_stacks = True
                    break

            if not block_in_stacks:
                logger.error("The block [%s] is not present in any of the stacks " % str(act.get_argument()))
                return self.get_world_state(), REWARD_INVALID, False
            
            # dynamics of the environment intervenes here: the pickup action may not succeed
            r = random.random()
            if r < self.dynamics:
                # with self.dynamics chance the block just remains on the table
                self.adata.holding = None
                self.adata.station = self.station
                self.adata.previous_action_succeeded = True
                reward = REWARD_VALID
            else:
                picked_up_block, r = self.world_state.pick_up(act.get_argument())
                if not picked_up_block:
                    return self.get_world_state(), REWARD_INVALID, False
                
                self.adata.holding = picked_up_block
                self.adata.station = self.station
                self.adata.previous_action_succeeded = True
                reward = REWARD_VALID

        elif act.get_type() == "putdown":
            # modify world;
            # current stack is always the last one
            current_stack = world_stacks[-1]
            if not self.adata.holding or self.adata.holding != act.get_argument():
                logger.error("The block [%s] is not in the agent arm " % str(act.get_argument()))
                return self.get_world_state(), REWARD_INVALID, False

            self.world_state.put_down(act.get_argument(), current_stack)

            self.adata.holding = None
            self.adata.previous_action_succeeded = True
            reward = REWARD_VALID

        elif act.get_type() == "unstack":
            block_in_stacks = False
            for stack in world_stacks:
                if act.get_first_arg() in stack:
                    block_in_stacks = True
                    break

            if not block_in_stacks:
                logger.error("The block [%s] is not in any of the current world stacks "
                                 % (str(act.get_first_arg())))
                return self.get_world_state(), REWARD_INVALID, False

            unstacked_block, r = self.world_state.unstack(act.get_first_arg(), act.get_second_arg())
            if not unstacked_block:
                return self.get_world_state(), REWARD_INVALID, False
            
            # environment dynamics
            r = random.random()
            if r < self.dynamics:
                # with self.dynamics chance the block gets dropped to the table
                current_stack = world_stacks[-1]
                self.world_state.put_down(unstacked_block, current_stack)

                self.adata.holding = None
                self.adata.previous_action_succeeded = True
                reward = REWARD_VALID
            else:
                self.adata.holding = unstacked_block
                self.adata.previous_action_succeeded = True
                reward = REWARD_VALID

        elif act.get_type() == "stack":
            block_in_stacks = False
            for stack in world_stacks:
                if act.get_second_arg() in stack:
                    block_in_stacks = True
                    break

            if not block_in_stacks:
                logger.error("The block [%s] is not in any of the current world stacks"
                                   % (str(act.get_second_arg())))
                return self.get_world_state(), REWARD_INVALID, False

            if not self.adata.holding or self.adata.holding != act.get_first_arg():
                return self.get_world_state(), REWARD_INVALID, False

            self.world_state.stack(act.get_first_arg(), act.get_second_arg())
            self.adata.holding = None
            self.adata.previous_action_succeeded = True
            reward = REWARD_VALID

        elif act.get_type() == "lock":
            block_in_stacks = False
            for stack in world_stacks:
                if act.get_argument() in stack:
                    block_in_stacks = True
                    break

            if not block_in_stacks:
                logger.error("The block [%s] is not in any of the current world stacks"
                                   % (str(act.get_argument())))
                return self.get_world_state(), REWARD_INVALID, False

            self.world_state.lock(act.get_argument())
            self.adata.previous_action_succeeded = True
            reward = REWARD_VALID

        elif act.get_type() == "no_action":
            self.adata.previous_action_succeeded = True
            reward = REWARD_VALID

        else:
            logger.error("Should not be here: action not recognized %s" % act.get_type())
            return self.get_world_state(), REWARD_INVALID, False

        target_predicates = [str(pred) for pred in self.target_state.to_predicates()]
        world_predicates = [str(pred) for pred in self.world_state.to_predicates()]

        completed = all(list(map(lambda pred: pred in world_predicates, target_predicates)))

        if completed:
            return self.get_world_state(), REWARD_FINAL, True

        return self.get_world_state(), reward, completed


    def get_world_state(self):
        world_state = self.world_state.get_state()
        
        # target_state = self.target_state.get_state()
        # world_state += ";" + target_state

        # if self.adata.holding:
        #     world_state += ";" + "holding(%s)" % (str(self.adata.holding))
        # else:
        #     world_state += ";" + "holding(none)"
        
        if self.adata.holding:
            world_state.append(self.adata.holding)
        else:
            world_state.append(Block("0"))
            
        return frozenset(world_state)
        

    def get_stacks_state(self):
        return self.world_state.get_state()

    def __str__(self):
        prefix = {}

        data = []
        data.append(" " + str(self.adata.agent))
        data.append(" <" + ("" if not self.adata.holding else str(self.adata.holding)) + ">")
        data.append("\n")
        prefix[self.world_state.get_stacks()[0]] = data

        suffix = {}

        data = ["====="]
        data.append(" " + str(self.station))
        suffix[self.world_state.get_stacks()[0]] = data

        return self.world_state._print_world(6, prefixes=prefix, suffixes=suffix, print_table=False)
