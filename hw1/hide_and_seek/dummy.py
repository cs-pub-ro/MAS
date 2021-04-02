from hide_and_seek.envs import HideAndSeekEnv, GameState, GameAction, Action, Reward
from hide_and_seek.envs import GridPosition, GridOrientation, GridRelativeOrientation
from random import choice
from time import sleep

class DummyAgent:
    ag_ids = [1, 2]
    
    def __init__(self):
        self.total_reward = 0

    def get_action(self, state: GameState):
        return {ag_id: choice([e for e in Action]) for ag_id in DummyAgent.ag_ids}


class DummyHider(DummyAgent):
    pass


class DummySeeker(DummyAgent):
    pass


if __name__ == "__main__":
    env = HideAndSeekEnv(max_turns=20)
    seeker_agent = DummySeeker()
    hider_agent = DummyHider()
    
    turn = 0
    state = env.reset()
    
    print("## State at turn: %i\n" % turn)
    print(env)
    
    while True:
        turn += 1
        seeker_actions = seeker_agent.get_action(state)
        hider_actions = hider_agent.get_action(state)
        
        game_action = GameAction(seeker_actions=seeker_actions, hider_actions=hider_actions)
        state, reward, done = env.step(action=game_action)
        
        seeker_agent.total_reward += reward.total_seeker_reward
        hider_agent.total_reward += reward.total_hider_reward
        
        print("## Turn: %i" % turn)
        print("\tSeeker actions: %s " % str(game_action.seeker_actions))
        print("\tSeeker reward: %6.2f" % seeker_agent.total_reward)

        print("\tHider actions: %s " % str(game_action.hider_actions))
        print("\tHider reward: %6.2f" % hider_agent.total_reward)
        
        print(env)
        
        sleep(0.5)

        if done:
            break

