import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Tuple

from hide_and_seek.envs import HideAndSeekEnv, GameState, GameAction, Action, Reward
from hide_and_seek.envs import GridPosition, GridOrientation, GridRelativeOrientation
from random import choice
from time import sleep
from .dummy import DummySeeker


class QLearningHider:
    """
    This agent learns how to play the hide and seek game by simulating random searching seekers
    """
    def __init__(self):
        # Q and rewards
        self.Q: Dict[Tuple[GameState, Tuple[Action, Action]], float] = {}
        self.rewards = []
        self.iterations = []
        
        self.dummy_seeker = DummySeeker()
        
    def learn(self, env: HideAndSeekEnv, alpha=0.75, discount=0.95, episodes=5000):
        # Episodes
        for episode in range(episodes):
            # Refresh state
            state = env.reset()
            done = False
            
            t_reward = 0
            max_steps = env.max_turns
        
            # Run episode
            for i in range(max_steps):
                if done:
                    break
            
                current = state
                seeker_actions = self.dummy_seeker.get_action(state)
                
                hider_actions = None
                best_action_val = None
                
                # get best action
                for act1, act2 in zip([e for e in Action], [e for e in Action]):
                    if not best_action_val:
                        hider_actions = act1, act2
                        best_action_val = self.Q.get((current, hider_actions), 0) \
                                          + np.random.randn(1, env.get_num_actions() * (1 / float(episode + 1)))
                    else:
                        action_val = self.Q.get((current, hider_actions), 0) \
                                          + np.random.randn(1, env.get_num_actions() * (1 / float(episode + 1)))
                        if action_val > best_action_val:
                            best_action_val = action_val
                            hider_actions = act1, act2
                
                game_action = GameAction(seeker_actions=seeker_actions,
                                         hider_actions={1: hider_actions[0], 2: hider_actions[1]})
                state, reward, done = env.step(game_action)
                t_reward += reward.total_hider_reward
                
                best_future_action = None
                best_future_val = None
                for act1, act2 in zip([e for e in Action], [e for e in Action]):
                    if not best_future_val:
                        best_future_val = self.Q.get((state, (act1, act2)), 0)
                    else:
                        future_diff = self.Q.get((state, (act1, act2)), 0)
                        if future_diff > best_future_val:
                            best_future_val = future_diff
                
                self.Q[(current, hider_actions)] += alpha * (reward.total_hider_reward + discount * best_future_val -
                                                             self.Q.get((current, hider_actions), 0))
                
            self.rewards.append(t_reward)
            self.iterations.append(i)
        
    def play(self, state: GameState):
        pass