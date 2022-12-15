''' Code that forms the environment in which the algorithms run. '''

# Open AI Imports
import gym
from gym.spaces import Discrete 

#  Standard lib imports
import numpy as np



class BanditEnv(gym.Env):
    ''' Bandit environments, to create an environment the user must
        pass in an array of rewards, and reward probabilites. '''

    # Metadata attribute
    metadata = {"render_modes": "human"}

    def __init__(self, rewards, reward_probas):
        self.num_of_bandits = len(rewards)
        self.action_space = Discrete(self.num_of_bandits)
        # We do not change between states in a bandit problem
        self.observation_space = Discrete(1)
        self.rewards = rewards
        self.reward_probas = reward_probas

        # Checking to see if they both match.
        if len(rewards) != len(reward_probas):
            raise Exception("Check that the rewards and the reward probabilites match")
    
    # Agent will select and arm, pull it, receive a reward. 
    def step(self, arm_pulled):
        ''' Returns the reward from the bandit selected, this function is called by the agent
            and will return the reward needed. '''

        # Check that the action is valid.
        if arm_pulled < 0 or arm_pulled > self.num_of_bandits:
            raise Exception("Action is not valid")
        
        # Setting the reward to the returned, and some extra information that must be returned by the step function (see gym docs).
        reward = self.rewards[arm_pulled] if np.random.random() < self.reward_probas[arm_pulled] else 0
        observation = 0
        done = False
        info = dict()

        return reward, observation, done, info



        

        