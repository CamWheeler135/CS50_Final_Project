''' Main file that will create the instances and call the functions to run the project. '''

# Standard Imports
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Local Imports
import plotting
from bandit_env import BanditEnv
from bandit_algos import RandomAgent, GreedyAgent, EpsilonGreedyAgent, UCBAgent

#---- Environment Set Up ----#

# Set the rewards and return probabilites for each bandit
rewards = np.array([100, 50, 2, 25, 40, 70, 10])
reward_probas = np.array([0.05, 0.6, 0.8, 0.45, 0.30, 0.45, 0.7])

# Create the environment
env = BanditEnv(rewards=rewards, reward_probas=reward_probas)

#--- Agent and Actions Set Up----#

# Create the agent
random_agent = RandomAgent(env=env, number_of_pulls=50000)
greedy_agent = GreedyAgent(env=env, number_of_pulls=50000)
egreedy_agent = EpsilonGreedyAgent(env=env, number_of_pulls=50000) # Epsilon default value = 0.2
ucb_agent = UCBAgent(env=env, number_of_pulls=50000)

# Tell the agent to perform its actions, this will return the rewards it got, the cumulative reward and the times it pulled each arm.
ra_performance = random_agent.perform_actions()
ga_performance = greedy_agent.perform_actions()
ega_performance = egreedy_agent.perform_actions()
ucb_performance = ucb_agent.perform_actions()


#---- Results and Plotting ----#

# Creates a directory to store the plots of the agents performance. 
plotting.create_directory("Plots")

print(f"Random Action Selection: {sum(ra_performance['rewards'])}")
print(f"Greedy Action Selection: {sum(ga_performance['rewards'])}")
print(f"Epsilon Greedy Action Selection: {sum(ega_performance['rewards'])}")
print(f"UCB Action Selection: {sum(ucb_performance['rewards'])}")


plotting.plot_all(ra_performance, agent_type="Random_Agent")
plotting.plot_all(performance=ga_performance, agent_type="Greedy_Agent")
plotting.plot_all(performance=ega_performance, agent_type="Epsilon_Greedy_Agent")
plotting.plot_all(performance=ucb_performance, agent_type="UCB_Agent")