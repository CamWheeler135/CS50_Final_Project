''' Main file that will create the instances and call the functions to run the project. '''

# Standard Imports
import numpy as np

# Local Imports
import plotting
from bandit_env import BanditEnv
from bandit_algos import RandomAgent

# Creates a directory to store the plots of the agents performance. 
plotting.create_directory("Plots")

# Set the rewards and return probabilites for each bandit
rewards = np.array([93, 10, 2, 25])
reward_probas = np.array([0.2, 0.9, 1, 0.45])

# Create the environment
env = BanditEnv(rewards=rewards, reward_probas=reward_probas)

# Create the agent
random_agent = RandomAgent(env=env, number_of_pulls=200)

# Tell the agent to perform its actions, this will return the rewards it got, the cumulative reward and the times it pulled each arm.
performace = random_agent.perform_actions()

print(f"Rewards = {performace['rewards']}")
print(f"Cumulative Reward = {performace['cumulative_rewards']}")
print(f"Arm Counter = {performace['arm_counter']}")
