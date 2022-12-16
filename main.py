''' Main file that will create the instances and call the functions to run the project. '''

# Standard Imports
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Local Imports
import plotting
from bandit_env import BanditEnv
from bandit_algos import RandomAgent, GreedyAgent

#---- Environment Set Up ----#

# Set the rewards and return probabilites for each bandit
rewards = np.array([93, 10, 2, 25])
reward_probas = np.array([0.2, 0.7, 0.8, 0.45])

# Create the environment
env = BanditEnv(rewards=rewards, reward_probas=reward_probas)

#--- Agent and Actions Set Up----#

# Create the agent
random_agent = RandomAgent(env=env, number_of_pulls=2000)
greedy_agent = GreedyAgent(env=env, number_of_pulls=2000)

# Tell the agent to perform its actions, this will return the rewards it got, the cumulative reward and the times it pulled each arm.
ra_performance = random_agent.perform_actions()
ga_performance = greedy_agent.perform_actions()


#---- Results and Plotting ----#

# Creates a directory to store the plots of the agents performance. 
plotting.create_directory("Plots")


print(f"Random Action Selection: {ra_performance['rewards']}")
print(f"Greedy Action Selection: {ga_performance['rewards']}")

plotting.cumulative_reward_plot(ra_performance, agent_type="Random", save_path=Path('Plots', "Random Agent Cumulative Reward"))
plotting.cumulative_reward_plot(ga_performance, agent_type="Greedy", save_path=Path('Plots', "Greedy Agent Cumulative Reward"))
plotting.actions_taken_plot(ra_performance, agent_type="Random", save_path=Path('Plots', "Random Agent Action Selection"))
plotting.actions_taken_plot(ga_performance, agent_type="Greedy", save_path=Path('Plots', "Greedy Agent Action Selection"))