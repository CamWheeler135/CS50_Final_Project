''' Main file that will create the instances and call the functions to run the project. '''

# Standard Imports
import numpy as np
from pathlib import Path

# Local Imports
import plotting
from bandit_env import BanditEnv
from bandit_algos import RandomAgent, GreedyAgent, EpsilonGreedyAgent, EGDecayAgent, UCBAgent, ThompsonSamplingAgent, SoftMaxAgent, OptimisticInitializationAgent

print("\nRunning Main.\n")

#---- Environment Set Up ----#

# Set the rewards and return probabilites for each bandit
number_of_bandits = 5
rewards = np.ones(number_of_bandits)
reward_probas = np.random.random(number_of_bandits)

# Create the environment
env = BanditEnv(rewards=rewards, reward_probas=reward_probas)

#---- Agent and Actions Set Up ----#

# Create the agent
random_agent = RandomAgent(env=env, number_of_pulls=50000)
greedy_agent = GreedyAgent(env=env, number_of_pulls=50000)
egreedy_agent = EpsilonGreedyAgent(env=env, number_of_pulls=50000) # Epsilon default value = 0.2.
egd_agent = EGDecayAgent(env=env, number_of_pulls=50000) # Epsilon default = 1, Epsilon decay rate default = 0.99, Epsilon min default = 0.1
ucb_agent = UCBAgent(env=env, number_of_pulls=50000) # Default c hyperparameter = 2.
ts_agent = ThompsonSamplingAgent(env=env, number_of_pulls=50000)
sm_agent = SoftMaxAgent(env=env, number_of_pulls=50000) # Default temperature value 0.01
oi_agent = OptimisticInitializationAgent(env=env, number_of_pulls=50000)

# Tell the agent to perform its actions, this returns a dictionary of metrics.
ra_performance = random_agent.perform_actions()
ga_performance = greedy_agent.perform_actions()
ega_performance = egreedy_agent.perform_actions()
egda_performance = egd_agent.perform_actions()
ucb_performance = ucb_agent.perform_actions()
ts_performance = ts_agent.perform_actions()
sm_performance = sm_agent.perform_actions()
oi_performance = oi_agent.perform_actions()

# #---- Results and Plotting ----#

print("\nFinished Running, Loading Metrics and Plots.\n")

print(f"Random Total Reward: {sum(ra_performance['rewards'])}")
print(f"Greedy Total Reward: {sum(ga_performance['rewards'])}")
print(f"Epsilon Greedy Total Reward: {sum(ega_performance['rewards'])}")
print(f"Epsilon Greedy Decay Total Reward: {sum(egda_performance['rewards'])}")
print(f"UCB Total Reward: {sum(ucb_performance['rewards'])}")
print(f"Thompson Sampling Total Reward: {sum(ts_performance['rewards'])}")
print(f"SoftmaxTotal Reward: {sum(sm_performance['rewards'])}")
print(f"Optimistic Initialization Reward: {sum(oi_performance['rewards'])}")

# Plotting all of the metrics from each agent separately.
plotting.plot_all_single(ra_performance, agent_type="Random_Agent")
plotting.plot_all_single(performance=ga_performance, agent_type="Greedy_Agent")
plotting.plot_all_single(performance=ega_performance, agent_type="Epsilon_Greedy_Agent")
plotting.plot_all_single(performance=egda_performance, agent_type="Epsilon_Greedy_with_Decay_Agent")
plotting.plot_epsilon_decay(performance=egda_performance, agent_type="Epsilon_Greedy_with_Decay_Agent")
plotting.plot_all_single(performance=ucb_performance, agent_type="UCB_Agent")
plotting.plot_all_single(performance=ts_performance, agent_type="Thompson_Sampling_Agent")
plotting.plot_all_single(performance=sm_performance, agent_type='Softmax_Agent')
plotting.plot_all_single(performance=oi_performance, agent_type="Optimistic_Initialization_Agent")

# Plotting comparative plots, comparing metrics from each agent together.
agents_performance = [ra_performance, ga_performance, ega_performance, egda_performance,
        ucb_performance, ts_performance, sm_performance, oi_performance]

names = ["Random Agent", 'Greedy Agent', 'epsilon-Greedy Agent', 'epsilon-Greedy Decay Agent', 
        'UCB Agent', 'Thompson Sampling Agent', 'Softmax Agent', 'Optimistic Initialization Agent']
plotting.cumulative_reward_compare_plot(agent_performance=agents_performance, names=names)
plotting.total_reward_comparison(agent_performance=agents_performance, names=names)
plotting.total_regret_comparison(agent_performance=agents_performance, names=names)

# agents_with_q = [ega_performance, ucb_performance, ga_performance]
# names = ['epsilon-Greedy', 'UCB', 'Greedy', 'True Values']
# plotting.action_value_comparison_plot(agent_performance=agents_with_q, names=names, true_vals=reward_probas)