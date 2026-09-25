from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import EzPickle

np.random.seed(0)
NC = 150
NS = 10
NT = 3
SCORE_LIM = 10
CHROMOSOMES = np.random.randint(-SCORE_LIM, SCORE_LIM, size=(NT, NC, NS))
SCORES = np.sum(CHROMOSOMES, axis=2)
TRAIT_NAMES = ['Diabetes', 'Asthma', 'Schizophrenia']

class IteratedMeioticSelection(gym.Env, EzPickle):
    """
    ## Description
    The IteratedMeioticSelection (IMS) environment is a MORL problem.
    From Iterated Meiotic Selection section of Methods for strong human germline engineering (https://berkeleygenomics.org/articles/Methods_for_strong_human_germline_engineering.html#iterated-meiotic-selection).

    ## Observation Space
    The observation space is a discrete 3D matrix of size [nT, nC, nS+2].
    With nC the (maximum) number of chromosomes, nS the number of segments, and nT the number of traits.
    The matrix represents the nC chromosomes, of nS segments, of nT traits, with each value the score of the segment on that trait.
    At nS+1 is the total score. At nS+2 is how many times that chromosome has been recombined.

    ## Action Space
    The actions is a discrete space of size (nC-1 * nC / 2) (choice=False) or (nC-1 * nC / 2 * nS-1) (choice=True). Each action is the recombination of the two chosen chromosomes, with mutations decreasing the score.
    New chromosomes are either added to the matrix (replacement="none"), replace the combined chromosomes (replacement="inplace"), or replace the lowest-scoring chromosomes (replacement="inplace_lowest")

    ## Reward Space
    The reward is nT dimensional, equal to the increase in the maximum score of each trait over the whole group of chromosomes. (objective="max")
    OR
    The reward is nT dimensional, equal to the increase in the score of trait over the chromosomes changed. (objective="diff")
    OR
    The reward is nT dimensional, equal to the increase in the score of trait relative to its previous mean and variance. (objective="mean")
    OR
    The reward is nT dimensional, equal to the increase in the variance of each trait over the whole group of chromosomes. (objective="var")
    (Idea: increased variance means separated score values means some small, some large means many large score variance instead of 1.)
    AND
    A large negative reward for increased generations (nT+2).

    ## Starting State
    A given group of chromosomes

    ## Termination
    The episode ends after MAX_TS=nS^3 steps

    ## Arguments
    - mut_rate: rate of mutation (mean, std)

    ## Credits
    Code by Rares Dobre-Baron
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode: Optional[str] = None, choice=False, replacement="none", objective="max", mut_rate=(-0.1, 0.1)):
        EzPickle.__init__(self, render_mode, choice, replacement, objective)

        self.render_mode = render_mode

        self.choice = choice
        self.replacement = replacement
        self.objective = objective
        self.mut_rate = mut_rate

        self.nS = NS
        self.nT = NT

        if self.replacement == "none":
            self.nC = NC * 3
            self._state = np.zeros((self.nT, self.nC, self.nS+2))
            self._state[:, :int(self.nC/3), :-2] = CHROMOSOMES
            self._state[:, :int(self.nC/3), -2] = SCORES
            self.current_size = NC
        else:
            self.nC = NC
            self._state = np.zeros((self.nT, self.nC, self.nS+2))
            self._state[:, :, :-2] = CHROMOSOMES
            self._state[:, :, -2] = SCORES
            self.current_size = self.nC
        
        self.observation_space = spaces.Box(low=-SCORE_LIM * self.nS, high=SCORE_LIM * self.nS ^ 3, shape=(self.nT, self.nC, self.nS+2), dtype=np.float32)

        if not self.choice:
            self.action_space = gym.spaces.Discrete(int((self.nC-1) * self.nC / 2.))
        else:
            self.action_space = gym.spaces.Discrete(int((self.nC-1) * self.nC / 2.) * (self.nS-1))

        self.reward_space = spaces.Box(low=-2 * SCORE_LIM * self.nS, high=2 * SCORE_LIM * self.nS ^ 3, shape=(self.nT+2,), dtype=np.float32)
        self.reward_dim = self.nT + 2 

        self.max_generation = 0

        self.MAX_TS = self.nS * self.nS * self.nS

        if self.objective == "diff" or self.objective == "mean":
            self.r_init = 0
        elif self.objective == "var":
            self.r_init = np.var(self._state[:, :, -2])
        else:
            self.r_init = np.max(self._state[:, :, -2])

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)

        if self.replacement == "none":
            self._state = np.zeros((self.nT, self.nC, self.nS+2))
            self._state[:, :int(self.nC/3), :-2] = CHROMOSOMES
            self._state[:, :int(self.nC/3), -2] = SCORES
        else:
            self._state = np.zeros((self.nT, self.nC, self.nS+2))
            self._state[:, :, :-2] = CHROMOSOMES
            self._state[:, :, -2] = SCORES

        self._timestep = 0
        if self.render_mode == "human":
            self.render()

        return self._state, {}

    def render(self):
        if self.render_mode == "human":
            return f"t={self._timestep}: {self._state[:, :, -2]}, \n with max: {np.max(self._state[:, :, -2])}"

    def step(self, action):
        rewards = np.zeros((self.nT+2,), dtype=np.float32)
        action_zone = action % ((self.nC-1) * self.nC / 2)

        # Prepare action
        # https://stackoverflow.com/questions/242711/algorithm-for-index-numbers-of-triangular-matrix-coefficients
        # https://en.wikipedia.org/wiki/Triangular_number
        # https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
        
        #k = [n*(n-1) / 2] - [(n-i) * (n-i-1) / 2] + (j - i - 1)
        n = self.nC
        k = action_zone
        g = n * (n - 1) / 2
        i = (n-1 - np.floor((np.sqrt(8*(g - k - 1)+1)+1)/2))
        #j = k + i + 1 - (n*(n-1) / 2) + ((n-i) * (n-i-1) / 2)
        j = k + i + 1 + (i*i - 2*n*i + i) / 2
        
        i = int(i)
        j = int(j)
        if i > self.current_size or j > self.current_size:
            i, j = i % self.current_size, j % self.current_size
        
        proper_action = (i, j)

        if self.choice:
            crossover_point = int(action / ((self.nC-1) * self.nC / 2))
        else:
            np.random.seed()
            crossover_point = np.random.randint(0, self.nS)
        
        # Perform crossover
        chr_1 = np.copy(self._state[:, proper_action[0], :-2])
        chr_2 = np.copy(self._state[:, proper_action[1], :-2])
        
        chr_1[:, :(crossover_point+1)], chr_2[:, :(crossover_point+1)] = np.copy(chr_2[:, :(crossover_point+1)]), np.copy(chr_1[:, :(crossover_point+1)])

        chr_1 = chr_1 + np.random.normal(self.mut_rate[0], self.mut_rate[1], chr_1.shape)
        chr_2 = chr_2 + np.random.normal(self.mut_rate[0], self.mut_rate[1], chr_2.shape)

        chr_1_sum = np.sum(chr_1, axis=1)
        chr_2_sum = np.sum(chr_2, axis=1)

        diff = 0
        mean_diff = 0

        # Change state
        if self.replacement == "none":
            if self.current_size < self.nC:
                diff = chr_1_sum - self._state[:, proper_action[0], -2] + \
                       chr_2_sum - self._state[:, proper_action[1], -2]
                mean_diff = (chr_1_sum + chr_2_sum - 2 * self._state[:, :self.current_size, -2].mean()) \
                            / (self._state[:, :self.current_size, -2].var())
                self._state[:, self.current_size, -1] = self._state[:, proper_action[0], -1] + 1
                self._state[:, self.current_size+1, -1] = self._state[:, proper_action[1], -1] + 1
                self._state[:, self.current_size, :-2] = chr_1
                self._state[:, self.current_size+1, :-2] = chr_2
                self._state[:, self.current_size, -2] = chr_1_sum
                self._state[:, self.current_size+1, -2] = chr_2_sum
                self.current_size += 2
            else:
                min_indices = np.argpartition(np.sum(self._state[:, :, -2], axis=0), kth=1)[:2]
                diff = chr_1_sum - self._state[:, min_indices[0], -2] + \
                       chr_2_sum - self._state[:, min_indices[1], -2]
                mean_diff = (chr_1_sum + chr_2_sum - 2 * self._state[:, :self.current_size, -2].mean()) \
                            / (self._state[:, :self.current_size, -2].var())
                self._state[:, min_indices[0], -1] = self._state[:, proper_action[0], -1] + 1
                self._state[:, min_indices[1], -1] = self._state[:, proper_action[1], -1] + 1
                self._state[:, min_indices[0], :-2] = chr_1
                self._state[:, min_indices[1], :-2] = chr_2
                self._state[:, min_indices[0], -2] = chr_1_sum
                self._state[:, min_indices[1], -2] = chr_2_sum
        else:
            if self.replacement == "inplace_lowest":
                min_indices = np.argpartition(np.sum(self._state[:, :, -2], axis=0), kth=1)[:2]
                diff = chr_1_sum - self._state[:, min_indices[0], -2] + \
                       chr_2_sum - self._state[:, min_indices[1], -2]
                mean_diff = (chr_1_sum + chr_2_sum - 2 * self._state[:, :self.current_size, -2].mean()) \
                            / (self._state[:, :self.current_size, -2].var())
                self._state[:, min_indices[0], -1] = self._state[:, proper_action[0], -1] + 1
                self._state[:, min_indices[1], -1] = self._state[:, proper_action[1], -1] + 1
                self._state[:, min_indices[0], :-2] = chr_1
                self._state[:, min_indices[1], :-2] = chr_2
                self._state[:, min_indices[0], -2] = chr_1_sum
                self._state[:, min_indices[1], -2] = chr_2_sum
            else:
                diff = chr_1_sum - self._state[:, proper_action[0], -2] + \
                       chr_2_sum - self._state[:, proper_action[1], -2]
                mean_diff = (chr_1_sum + chr_2_sum - 2 * self._state[:, :self.current_size, -2].mean()) \
                        / (self._state[:, :self.current_size, -2].var())
                self._state[:, proper_action[0], -1] += 1
                self._state[:, proper_action[1], -1] += 1
                self._state[:, proper_action[0], :-2] = chr_1
                self._state[:, proper_action[1], :-2] = chr_2
                self._state[:, proper_action[0], -2] = chr_1_sum
                self._state[:, proper_action[1], -2] = chr_2_sum
        
        # Calculate reward
        rewards[-1] = -0.01
        if self.max_generation < np.max(self._state[:, :, -1]):
            rewards[-2] = -10
            self.max_generation = np.max(self._state[:, :, -1])
        else:
            rewards[-2] = 0
        
        if self.objective == "diff":
            rewards[:-2] = diff
        elif self.objective == "mean":
            rewards[:-2] = mean_diff
        elif self.objective == "var":
            rewards[:-2] = np.var(self._state[:, :, -2]) - self.r_init
            self.r_init = np.var(self._state[:, :, -2])
        else:
            rewards[:-2] = np.max(self._state[:, :, -2]) - self.r_init
            self.r_init = np.max(self._state[:, :, -2])
        
        # Ending
        self._timestep += 1
        if self.render_mode == "human":
            self.render()
        return self._state, rewards, self._timestep == self.MAX_TS, self._timestep == self.MAX_TS, {}


if __name__ == "__main__":
    env = IteratedMeioticSelection()
    terminated = False
    env.reset()
    while True:
        env.render()
        action_sample = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action_sample)
        if terminated:
            env.reset()
