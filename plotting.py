''' Code that plots the performance of the algorithm. '''
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="darkgrid")

def create_directory(directory_path: str):
    ''' Makes a folder in the directory to place all of the plots. '''
    
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def cumulative_reward_plot(performance: dict, agent_type: str):
    fig, ax = plt.subplots()
    ax.plot(performance["cumulative_rewards"])
    ax.set_ylabel("Cumulative Reward")
    ax.set_xlabel("Time Steps")
    ax.set_title(f"{agent_type} Performance")
    plt.savefig(Path("Plots/" + agent_type + 'cumulative_reward'))


def actions_taken_plot(performance: dict, agent_type: str):
    ''' Plots a bar chart of the times each action was taken. '''

    x = [i for i in range(len(performance["arm_counter"]))]
    y = performance["arm_counter"]

    fig, ax = plt.subplots()
    ax.bar(x, y, width=0.5)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_ylabel("Times Chosen")
    ax.set_xlabel("Bandit Choice")
    ax.set_title(f"{agent_type} Action Selection")
    plt.savefig(Path("Plots/" + agent_type + 'arm_choice'))


def plot_all(performance: dict, agent_type: str):
    ''' Calls all of the plotting functions. '''
    cumulative_reward_plot(performance=performance, agent_type=agent_type)
    actions_taken_plot(performance=performance, agent_type=agent_type)