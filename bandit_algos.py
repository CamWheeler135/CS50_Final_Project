''' Classes of the different multi-armed bandit algorithms. '''
import numpy as np

class RandomAgent():
    ''' An agent that select actions at random.
        Initialized with the environment and 
        number of interactions we want the agent to have. '''

    def __init__(self, env, number_of_pulls):
        self.env = env
        self.number_of_pulls = number_of_pulls

        # Internal states of the agent
        self.rewards = []
        self.cumulative_reward = []
        self.pulled_arm_counter = np.zeros(self.env.num_of_bandits)

    def perform_actions(self):

        for i in range(self.number_of_pulls):
            selected_action = np.random.choice(self.env.num_of_bandits)
            reward, _, _, _ = self.env.step(selected_action)
            self.rewards.append(reward)
            self.cumulative_reward.append(sum(self.rewards) / len(self.rewards))
            self.pulled_arm_counter[selected_action] += 1
        
        return {"rewards": self.rewards, "cumulative_rewards": self.cumulative_reward, "arm_counter": self.pulled_arm_counter}

