import numpy as np
from typing import Dict, Tuple, List
from enum import IntEnum


class Actions(IntEnum):
    LEFT = 0
    RIGHT = 1
    NOOP = 2


class States(IntEnum):
    ###########################
    # S0 #  S1 # S2 # S3 # S4 #
    ###########################
    S0 = 0
    S1 = 1
    S2 = 2 # the goal should be here
    S3 = 3
    S4 = 4


class Obs(IntEnum):
    # The observation is the number of walls
    O2 = 0
    O3 = 1


class MyEnv(object):
    def __init__(self, max_num_steps: int = 2):
        """
        Constructor

        Parameters
        ----------
        max_num_steps
            maximum number of steps allowed in the env
        """
        self.max_num_steps = max_num_steps
        self.num_steps = None
        self.__state = None
        self.done = True

        # define state mapping
        self.__num_states = 2
        self.__state_mapping = {
            States.S0: "S0",
            States.S1: "S1",
            States.S2: "S2",
            States.S3: "S3",
            States.S4: "S4",
        }

        # define action mapping
        self.__num_actions = 3
        self.__action_mapping = {
            Actions.LEFT: "Left",
            Actions.RIGHT: "Right",
            Actions.NOOP: "NoOp"
        }

        # define observation mapping
        self.__num_obs = 2
        self.__obs_mapping = {
            Obs.O2: "Two walls",
            Obs.O3: "Three walls",
        }

        # init transitions & observations probabilities
        # and rewards
        self.__init_transitions()
        self.__init_observations()
        self.__init_rewards()

    def __init_transitions(self):


        # define transition probability for left action
        #         S0    S1     S2     S3     S4
        #  S0     1.0   0.0    0.0    0.0    0.0
        #  S1     1.0   0.0    0.0    0.0    0.0
        #  S2     0.0   1.0    0.0    0.0    0.0
        #  S3     0.0   0.0    1.0    0.0    0.0
        #  S4     0.0   0.0    0.0    1.0    0.0

        _left = np.array([[
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0]
        ]])

        # define transition probability for right action
        #         S0    S1     S2     S3     S4
        #  S0     0.0   1.0    0.0    0.0    0.0
        #  S1     0.0   0.0    1.0    0.0    0.0
        #  S2     0.0   0.0    0.0    1.0    0.0
        #  S3     0.0   0.0    0.0    0.0    1.0
        #  S4     0.0   0.0    0.0    0.0    1.0
        _right = np.array([[
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ]])

        # define transition probability for left action
        #         S0    S1     S2     S3     S4
        #  S0     1.0   0.0    0.0    0.0    0.0
        #  S1     0.0   1.0    0.0    0.0    0.0
        #  S2     0.0   0.0    1.0    0.0    0.0
        #  S3     0.0   0.0    0.0    1.0    0.0
        #  S4     0.0   0.0    0.0    0.0    1.0

        _noop = np.array([[
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ]])

        self.__T = np.concatenate([_left, _right, _noop], axis=0)

    def __init_observations(self):
        # define observation probability for the left action
        #      Obs:2    Obs:3
        # S0   0.0      1.0
        # S1   0.0      1.0
        # S2   1.0      0.0
        # S3   1.0      0.0
        # S4   1.0      0.0
        O_left = np.array([[
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0]
        ]])

        # define observation probability for the right action
        #      Obs:2    Obs:3
        # S0   1.0      0.0
        # S1   1.0      0.0
        # S2   1.0      0.0
        # S3   0.0      1.0
        # S4   0.0      1.0
        O_right = np.array([[
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0]
        ]])

        # define observation probability for the noop action
        #      Obs:2    Obs:3
        # S0   0.0      1.0
        # S1   1.0      0.0
        # S2   1.0      0.0
        # S3   1.0      0.0
        # S4   0.0      1.0
        O_noop = np.array([[
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ]])

        self.__O = np.concatenate([O_left, O_right, O_noop], axis=0)

    def __init_rewards(self):
        # define rewards for left action
        # S0: 0
        # S1: 0
        # S2: 0
        # S3: 1
        # S4: 0
        R_left = np.array([[0, 0, 0, 1, 0]])

        # define rewards for right action
        # S0: 0
        # S1: 1
        # S2: 0
        # S3: 0
        # S4: 0
        R_right = np.array([[0, 1, 0, 0, 0]])

        # define rewards for noop action
        # S0: 0
        # S1: 1
        # S2: 0
        # S3: 0
        # S4: 0
        R_noop = np.array([[0, 0, 1, 0, 0]])
        self.__R = np.concatenate([R_left, R_right, R_noop], axis=0)

    def reset(self):
        self.done = False
        self.num_steps = 0

        # initialize the state random
        # this puts the tiger behind the left and right
        # door with equal probability
        self.__state = np.random.choice([States.S0, States.S4])

    def step(self, action: Actions) -> Tuple[int, float, bool, Dict[str, int]]:
        """
        Performs an environment step

        Parameters
        ----------
        action
            action to be applied

        Returns
        -------
        Tuple containing the next observation, the reward,
        ending episode flag, other information.
        """
        assert not self.done, "The episode finished. Call reset()!"
        self.num_steps += 1
        self.done = (self.num_steps == self.max_num_steps)

        # get the next observation. this is stochastic
        obs = np.random.choice(
            a=[Obs.O2, Obs.O3],
            p=self.O[action][self.__state]
        )

        # get the reward. this is deterministic
        reward = self.R[action][self.__state]

        # get the next transition
        self.__state = np.random.choice(
            a=list(States),
            p=self.T[action][self.__state]
        )

        # construct info
        info = {"num_steps": self.num_steps}
        return obs, reward, self.done, info

    @property
    def state_mapping(self) -> Dict[States, str]:
        """
        Returns
        -------
        State mapping (for display purpose)
        """
        return self.__state_mapping

    @property
    def action_mapping(self) -> Dict[Actions, str]:
        """
        Returns
        -------
        Action mapping (for display purposes)
        """
        return self.__action_mapping

    @property
    def obs_mapping(self) -> Dict[Obs, str]:
        """
        Returns
        -------
        Observation mapping (for display purposes)
        """
        return self.__obs_mapping

    @property
    def T(self) -> np.ndarray:
        """
        Returns
        -------
        Transition probability matrix.
        Axes: (a, s, s')
        """
        return self.__T

    @property
    def O(self) -> np.ndarray:
        """
        Returns
        -------
        Observation probability matrix.
        Axes: (a, s, o)
        """
        return self.__O

    @property
    def R(self) -> np.ndarray:
        """
        Returns
        -------
        Reward matrix:
        Axes: (a, s)
        """
        return self.__R

    @property
    def states(self) -> List[int]:
        """
        Returns
        -------
        List containing the states
        """
        return list(self.__state_mapping.keys())

    @property
    def actions(self) -> List[int]:
        """
        Returns
        -------
        List containing the actions
        """
        return list(self.__action_mapping.keys())

    @property
    def obs(self) -> List[int]:
        """
        Returns
        -------
        List containing the observations
        """
        return list(self.__obs_mapping.keys())