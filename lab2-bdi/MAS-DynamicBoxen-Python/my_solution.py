from environment import *
from typing import Dict, List, Optional
import time

from my import MyAgent, PlaceBlockDesire


class PlaceAndLockDesire(PlaceBlockDesire):
    """
    Concrete desire used by the solution strategy:
    the block must be in its target support relation AND locked.
    """

    def is_achieved(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        if not super().is_achieved(current_world, holding_block):
            return False

        try:
            return current_world.get_stack(self.block).is_locked(self.block)
        except Exception:
            return False

    def is_impossible(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        try:
            current_world.get_stack(self.block)
        except Exception:
            return holding_block != self.block

        if self.support is not None:
            try:
                current_world.get_stack(self.support)
            except Exception:
                return holding_block != self.support

        return False


class MySolutionAgent(MyAgent):
    """
    Implements the requested strategy:
      - desires are of type "put block in correct place"
      - for each desire use unstack-stack planning:
          clear above block -> clear above support -> pickup/unstack block -> place -> lock
      - desire ordering by stacks:
          1) smaller target stacks first
          2) stacks solvable by shuffling only internal blocks first
          3) stacks already more than half correct first
    """

    def __init__(self, name: str, target_state: BlocksWorld):
        self._target_stacks: List[List[Block]] = []
        self._target_support: Dict[Block, Optional[Block]] = {}
        self.max_wait_steps: int = 10
        self._wait_steps_left: int = self.max_wait_steps
        super(MySolutionAgent, self).__init__(name=name, target_state=target_state)

    def _initialize_desire_pool_from_target(self) -> None:
        self.desire_pool = []
        self._target_stacks = []
        self._target_support = {}

        for stack in self.target_state.get_stacks():
            blocks = list(stack.get_blocks())
            self._target_stacks.append(blocks)

            for idx, block in enumerate(blocks):
                support = None if idx == 0 else blocks[idx - 1]
                self._target_support[block] = support
                self.desire_pool.append(PlaceAndLockDesire(block=block, support=support))

    def response(self, perception: BlocksWorldPerception) -> BlocksWorldAction:
        act = super(MySolutionAgent, self).response(perception)

        if act.get_type() != "no_action":
            self._wait_steps_left = self.max_wait_steps
            return act

        if perception.holding_block is not None:
            self.last_failure_reason = "fallback: no desire selected while holding a block; put it down"
            self.last_action = PutDown(perception.holding_block)
            self._wait_steps_left = self.max_wait_steps
            return self.last_action

        remaining_desires = [
            desire
            for desire in self.desire_pool
            if isinstance(desire, PlaceAndLockDesire) and not desire.is_achieved(perception.current_world, perception.holding_block)
        ]

        if len(remaining_desires) > 0 and all(
                desire.is_impossible(perception.current_world, perception.holding_block) for desire in remaining_desires):
            if self._wait_steps_left > 0:
                self._wait_steps_left -= 1
                self.last_failure_reason = (
                    f"all desires currently impossible; waiting for hidden blocks (remaining wait steps={self._wait_steps_left})"
                )
                self.last_action = NoAction()
                return self.last_action

            self.last_failure_reason = "all desires remained impossible after wait budget; stopping"
            self.last_action = AgentCompleted()
            return self.last_action

        self._wait_steps_left = self.max_wait_steps

        return act

    def _select_next_desire(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> Optional[PlaceAndLockDesire]:
        desire_by_block: Dict[Block, PlaceAndLockDesire] = {
            desire.block: desire
            for desire in self.desire_pool
            if isinstance(desire, PlaceAndLockDesire)
        }

        ordered_stacks = sorted(
            self._target_stacks,
            key=lambda target_stack: (
                len(target_stack),
                0 if self._requires_only_internal_shuffling(target_stack, current_world) else 1,
                0 if self._is_more_than_half_correct(target_stack, current_world) else 1,
            )
        )

        for target_stack in ordered_stacks:
            for block in target_stack:
                desire = desire_by_block.get(block)
                if desire is None:
                    continue

                if desire.is_achieved(current_world, holding_block):
                    continue

                if desire.is_impossible(current_world, holding_block):
                    continue

                # Skip desires whose support block exists in the target stack
                # but is not yet locked — we must lock bottom blocks first.
                support = self._target_support.get(block)
                if support is not None:
                    try:
                        support_stack = current_world.get_stack(support)
                        if not support_stack.is_locked(support):
                            continue
                    except Exception:
                        continue

                return desire

        return None

    def _plan_for_current_desire(self, current_world: BlocksWorld, holding_block: Optional[Block]) -> List[BlocksWorldAction]:
        if self.current_desire is None or not isinstance(self.current_desire, PlaceAndLockDesire):
            return []

        desire = self.current_desire
        if desire.is_achieved(current_world, holding_block):
            return []

        actions: List[BlocksWorldAction] = []
        sim_world = current_world.clone()
        sim_holding = holding_block

        # Fast path: if strict lock preconditions are satisfied already, lock directly.
        if sim_holding is None:
            try:
                if self._can_lock_strict(desire.block, sim_world):
                    actions.append(Lock(desire.block))
                    return actions
            except Exception:
                pass

        # If block is already placed correctly, only need to clear above it and lock.
        # Do not pick it up and re-place — that would unnecessarily disturb the stack.
        block_already_placed = PlaceBlockDesire.is_achieved(desire, sim_world, sim_holding)
        if block_already_placed:
            if sim_holding is not None:
                actions.append(PutDown(sim_holding))
                sim_world.put_down(sim_holding, sim_world.get_stacks()[-1])
                sim_holding = None
            sim_holding = self._clear_above(desire.block, sim_world, actions, sim_holding)
            if not self._can_lock_strict(desire.block, sim_world):
                return []
            actions.append(Lock(desire.block))
            return actions

        if sim_holding is not None and sim_holding != desire.block:
            actions.append(PutDown(sim_holding))
            sim_world.put_down(sim_holding, sim_world.get_stacks()[-1])
            sim_holding = None

        if sim_holding == desire.block and desire.support is not None:
            actions.append(PutDown(desire.block))
            sim_world.put_down(desire.block, sim_world.get_stacks()[-1])
            sim_holding = None

        sim_holding = self._clear_above(desire.block, sim_world, actions, sim_holding)

        if desire.support is not None:
            sim_holding = self._clear_above(desire.support, sim_world, actions, sim_holding)

        if sim_holding != desire.block:
            try:
                block_stack = sim_world.get_stack(desire.block)
            except Exception:
                return []

            if block_stack.is_on_table(desire.block):
                actions.append(PickUp(desire.block))
                sim_world.pick_up(desire.block)
            else:
                below = block_stack.get_below(desire.block)
                actions.append(Unstack(desire.block, below))
                sim_world.unstack(desire.block, below)
            sim_holding = desire.block

        if desire.support is None:
            actions.append(PutDown(desire.block))
            sim_world.put_down(desire.block, sim_world.get_stacks()[-1])
        else:
            actions.append(Stack(desire.block, desire.support))
            sim_world.stack(desire.block, desire.support)
        sim_holding = None

        if not self._can_lock_strict(desire.block, sim_world):
            return []

        actions.append(Lock(desire.block))
        return actions

    def _target_prefix_to_block(self, block: Block) -> List[Block]:
        for target_stack in self._target_stacks:
            if block in target_stack:
                idx = target_stack.index(block)
                return list(target_stack[:idx + 1])
        return []

    def _is_target_prefix_aligned(self, block: Block, current_world: BlocksWorld) -> bool:
        prefix = self._target_prefix_to_block(block)
        if len(prefix) == 0:
            return False

        for idx, current_block in enumerate(prefix):
            try:
                stack = current_world.get_stack(current_block)
            except Exception:
                return False

            if idx == 0:
                if not stack.is_on_table(current_block):
                    return False
            else:
                if not stack.is_on(current_block, prefix[idx - 1]):
                    return False

        return True

    def _are_prefix_supports_locked(self, block: Block, current_world: BlocksWorld) -> bool:
        prefix = self._target_prefix_to_block(block)
        if len(prefix) <= 1:
            return True

        for support_block in prefix[:-1]:
            try:
                support_stack = current_world.get_stack(support_block)
                if not support_stack.is_locked(support_block):
                    return False
            except Exception:
                return False

        return True

    def _can_lock_strict(self, block: Block, current_world: BlocksWorld) -> bool:
        if not self._is_target_prefix_aligned(block, current_world):
            return False

        if not self._are_prefix_supports_locked(block, current_world):
            return False

        try:
            stack = current_world.get_stack(block)
            if stack.is_locked(block):
                return False
            if not stack.is_clear(block):
                return False
        except Exception:
            return False

        return True

    def _requires_only_internal_shuffling(self, target_stack: List[Block], current_world: BlocksWorld) -> bool:
        target_set = set(target_stack)

        anchor_stack = None
        for block in target_stack:
            try:
                stack = current_world.get_stack(block)
            except Exception:
                return False

            if anchor_stack is None:
                anchor_stack = stack
            elif stack != anchor_stack:
                return False

        if anchor_stack is None:
            return False

        return set(anchor_stack.get_blocks()) == target_set

    def _is_more_than_half_correct(self, target_stack: List[Block], current_world: BlocksWorld) -> bool:
        correct = 0
        for idx, block in enumerate(target_stack):
            try:
                stack = current_world.get_stack(block)
                if idx == 0:
                    if stack.is_on_table(block):
                        correct += 1
                else:
                    if stack.is_on(block, target_stack[idx - 1]):
                        correct += 1
            except Exception:
                pass

        return correct > (len(target_stack) / 2.0)

    def _clear_above(
        self,
        base_block: Block,
        sim_world: BlocksWorld,
        actions: List[BlocksWorldAction],
        sim_holding: Optional[Block],
    ) -> Optional[Block]:

        if sim_holding is not None:
            actions.append(PutDown(sim_holding))
            sim_world.put_down(sim_holding, sim_world.get_stacks()[-1])
            sim_holding = None

        while True:
            try:
                stack = sim_world.get_stack(base_block)
            except Exception:
                return sim_holding

            top_block = stack.get_top_block()
            if top_block == base_block:
                return sim_holding

            if stack.is_locked(top_block):
                return sim_holding

            below = stack.get_below(top_block)
            if below is None:
                return sim_holding

            actions.append(Unstack(top_block, below))
            sim_world.unstack(top_block, below)
            sim_holding = top_block

            actions.append(PutDown(top_block))
            sim_world.put_down(top_block, sim_world.get_stacks()[-1])
            sim_holding = None


class Tester(object):
    STEP_DELAY = 0.0
    TEST_SUITE = "tests/0e-large/"
    VERBOSE = True

    EXT = ".txt"
    SI = "si"
    SF = "sf"

    DYNAMICS_PROB = 0.75
    MAX_WAIT_STEPS = 100

    AGENT_NAME = "*A"

    def __init__(self):
        self._environment = None
        self._agents = []

        self._initialize_environment(Tester.TEST_SUITE)
        self._initialize_agents(Tester.TEST_SUITE)

    def _initialize_environment(self, test_suite: str) -> None:
        filename = test_suite + Tester.SI + Tester.EXT

        with open(filename) as input_stream:
            self._environment = DynamicEnvironment(
                BlocksWorld(input_stream=input_stream),
                verbose=Tester.VERBOSE,
                dynamics_prob=Tester.DYNAMICS_PROB,
            )

    def _initialize_agents(self, test_suite: str) -> None:
        filename = test_suite + Tester.SF + Tester.EXT

        with open(filename) as input_stream:
            desires = BlocksWorld(input_stream=input_stream)
            agent = MySolutionAgent(Tester.AGENT_NAME, desires)
            agent.max_wait_steps = Tester.MAX_WAIT_STEPS
            agent._wait_steps_left = agent.max_wait_steps
            self._agents.append(agent)

            self._environment.add_agent(agent, desires, None)

            if Tester.VERBOSE:
                print("Agent %s desires:" % str(agent))
                print(str(desires))

    def make_steps(self):
        if Tester.VERBOSE:
            print("\n\n================================================= INITIAL STATE:")
            print(str(self._environment))
            print("\n\n=================================================")
        else:
            print("Simulation started (verbose=False)")

        completed = False
        nr_steps = 0

        while not completed:
            completed = self._environment.step()

            time.sleep(Tester.STEP_DELAY)
            if Tester.VERBOSE:
                print(str(self._environment))

                for ag in self._agents:
                    print(ag.status_string())

            nr_steps += 1

            if Tester.VERBOSE:
                print("\n\n================================================= STEP %i completed." % nr_steps)

        if Tester.VERBOSE:
            print("\n\n================================================= ALL STEPS COMPLETED")
        else:
            print("Simulation completed in %i steps" % nr_steps)


if __name__ == "__main__":
    tester = Tester()
    tester.make_steps()
