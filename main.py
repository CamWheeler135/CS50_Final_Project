''' Main file that will create the instances and call the functions to run the project. '''

# Standard Imports
import numpy as np
from pathlib import Path

# Local Imports
import plotting
from bandit_env import BanditEnv
from bandit_algos import RandomAgent, GreedyAgent, EpsilonGreedyAgent, UCBAgent, ThompsonSamplingAgent

print("\nRunning Main.\n")

#---- Environment Set Up ----#

# Set the rewards and return probabilites for each bandit
rewards = np.array([1, 1, 1, 1, 1, 1, 1])
reward_probas = np.array([0.6, 0.3, 0.5, 0.7, 0.2, 0.8, 0.3])

# Create the environment
env = BanditEnv(rewards=rewards, reward_probas=reward_probas)

#---- Agent and Actions Set Up ----#

# Create the agent
random_agent = RandomAgent(env=env, number_of_pulls=100000)
greedy_agent = GreedyAgent(env=env, number_of_pulls=100000)
egreedy_agent = EpsilonGreedyAgent(env=env, number_of_pulls=100000) # Epsilon default value = 0.2.
ucb_agent = UCBAgent(env=env, number_of_pulls=100000) # Default c hyperparameter = 2.
ts_agent = ThompsonSamplingAgent(env=env, number_of_pulls=100000)

# Tell the agent to perform its actions, this returns a dictionary of metrics.
ra_performance = random_agent.perform_actions()
ga_performance = greedy_agent.perform_actions()
ega_performance = egreedy_agent.perform_actions()
ucb_performance = ucb_agent.perform_actions()
ts_performance = ts_agent.perform_actions()


#---- Results and Plotting ----#

# Creates a directory to store the plots of the agents performance.

print("\nFinished Running, Loading Metrics and Plots.\n")

print(f"Random Total Reward: {sum(ra_performance['rewards'])}")
print(f"Greedy Total Reward: {sum(ga_performance['rewards'])}")
print(f"Epsilon Greedy Total Reward: {sum(ega_performance['rewards'])}")
print(f"UCB Total Reward: {sum(ucb_performance['rewards'])}")
print(f"Thompson Sampling Total Reward: {sum(ts_performance['rewards'])}")

# Plotting all of the metrics from each agent separately.
plotting.plot_all_single(ra_performance, agent_type="Random_Agent")
plotting.plot_all_single(performance=ga_performance, agent_type="Greedy_Agent")
plotting.plot_all_single(performance=ega_performance, agent_type="Epsilon_Greedy_Agent")
plotting.plot_all_single(performance=ucb_performance, agent_type="UCB_Agent")
plotting.plot_all_single(performance=ts_performance, agent_type="Thompson_Sampling_Agent")

# Plotting comparative plots, comparing metrics from each agent together.
agents_performance = [ra_performance, ga_performance, ega_performance, ucb_performance, ts_performance]
names = ["Random Actions", 'Greedy Actions', 'epsilon-Greedy Actions', 'UCB', 'Thompson Sampling']
plotting.cumulative_reward_compare_plot(agent_performance=agents_performance, names=names)

agents_with_q = [ega_performance, ucb_performance, ga_performance]
names = ['epsilon-Greedy', 'UCB', 'Greedy', 'True Values']
plotting.action_value_comparison_plot(agent_performance=agents_with_q, names=names, true_vals=reward_probas)