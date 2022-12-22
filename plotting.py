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
    create_directory(Path("Plots/" + agent_type + "/"))
    plt.savefig(Path("Plots/" + agent_type + "/" + 'cumulative_reward'))


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
    plt.savefig(Path("Plots/" + agent_type + "/" +'arm_choice'))

def act_val_estimate_plot(performance: dict, agent_type: str):
    ''' Plots the action value estimate Q(a) for algorithms that compute it. '''
    x = [i for i in range(len(performance["q_values"]))]
    y = performance["q_values"]

    fig, ax = plt.subplots()
    ax.bar(x, y, width=0.5)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_ylabel("Action Value Estimate")
    ax.set_xlabel("Bandit")
    ax.set_title(f"{agent_type} Estimated Q(a) Value")
    plt.savefig(Path("Plots/" + agent_type + "/" + 'action_value_estimate'))


def total_regret_plot(performance: dict, agent_type: str):
    ''' Plots the regret of each algorithm. '''
    
    fig, ax = plt.subplots()
    ax.plot(performance['regret'])
    ax.set_title("Regret")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Total Regret")
    plt.savefig(Path("Plots/" + agent_type + "/" + "total_regret"))


def posterior_dists_plot(performance: dict, agent_type="Thompson Sampling"):
    ''' Plots the posterior beta distributions of each distribution for each bandit. '''
    
    fig, ax = plt.subplots()

    number_of_bandits = len(performance['arm_counter'])

    for beta_dists in range(number_of_bandits):
        alpha = int(performance['posterior_dists'][0, beta_dists])
        beta = int(performance['posterior_dists'][1, beta_dists])
        samples = [np.random.beta(a=alpha, b=beta) for i in range(10000)]
        sns.kdeplot(samples, fill=True)
    
    ax.legend(["Bandit %s" %(beta_dists) for beta_dists in range(number_of_bandits)], loc="upper left")
    ax.set_xlabel("Mean Reward")


    plt.savefig(Path("Plots/" + agent_type + "/" + "Posterior_distributions"))



def plot_all(performance: dict, agent_type: str):
    ''' Calls all of the plotting functions. '''

    if agent_type == 'Random_Agent':
        cumulative_reward_plot(performance=performance, agent_type=agent_type)
        actions_taken_plot(performance=performance, agent_type=agent_type)
        total_regret_plot(performance=performance, agent_type=agent_type)

    elif agent_type == 'Thompson_Sampling_Agent':
        cumulative_reward_plot(performance=performance, agent_type=agent_type)
        actions_taken_plot(performance=performance, agent_type=agent_type)
        total_regret_plot(performance=performance, agent_type=agent_type)
        posterior_dists_plot(performance=performance, agent_type=agent_type)

    else:
        cumulative_reward_plot(performance=performance, agent_type=agent_type)
        actions_taken_plot(performance=performance, agent_type=agent_type)
        act_val_estimate_plot(performance=performance, agent_type=agent_type)
        total_regret_plot(performance=performance, agent_type=agent_type)
