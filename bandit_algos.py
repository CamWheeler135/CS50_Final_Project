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
        ''' For each timestep, the agent will select and action. '''

        for i in range(self.number_of_pulls):
            selected_action = np.random.choice(self.env.num_of_bandits)
            reward, _, _, _ = self.env.step(selected_action)
            self.rewards.append(reward)
            self.cumulative_reward.append(sum(self.rewards) / len(self.rewards))
            self.pulled_arm_counter[selected_action] += 1
        
        return {"rewards": sum(self.rewards), "cumulative_rewards": self.cumulative_reward, "arm_counter": self.pulled_arm_counter}


class GreedyAgent():
    ''' An agent that selects the action with the highest
        action value 100% of the time. '''

    def __init__(self, env, number_of_pulls):
        self.env = env
        self.number_of_pulls = number_of_pulls
        
        # Internal states of the agent, note that we also have to add the notion of an "action value" (q_value)
        self.rewards = []
        self.cumulative_reward = []
        self.pulled_arm_counter = np.zeros(self.env.num_of_bandits)
        self.Q_values = np.zeros(self.env.num_of_bandits)

    def update_Q(self, selected_action, reward, action_count):
        ''' Updates the Q value for the action by averaging the immediate rewards
            over the times that action has been selected. '''

        updated_value = (self.Q_values[selected_action] + reward) / action_count[selected_action]
        self.Q_values[selected_action] = updated_value

    def select_greedy_action(self, q_values):
        ''' Selects an bandit to pull, if q values for any bandits are equal
            function will break ties randomly. Returning the action to be selected. '''

        # If any of the q_values are equal, select one of those values at random. 
        arg_max = np.argmax(q_values) 
        possible_actions = np.where(q_values == q_values[arg_max]) # Returns an array of indices of possible actions (if there is a tie there will be more than 1 element in the array.)

        # If the length of the array = 1 then we have a greedy action to select.
        if len(possible_actions[0]) == 1:
            return arg_max
        else:
            # In the case of ties, choose on of those actions randomly.  
            selected_action = np.random.choice(possible_actions[0])
            return selected_action


    def perform_actions(self):
        ''' For each timestep the agent will select an action according
            according to the highest action value (q_value) '''

        for i in range(self.number_of_pulls):

            # We need to tell the agent how to make decisions if action values are equal.
            selected_action = self.select_greedy_action(q_values=self.Q_values)
            reward, _, _, _ = self.env.step(selected_action)
            self.rewards.append(reward)
            self.cumulative_reward.append(sum(self.rewards) / len(self.rewards))
            self.pulled_arm_counter[selected_action] += 1

            # Update the Q value of that action.
            self.update_Q(selected_action=selected_action, reward=reward, action_count=self.pulled_arm_counter)


        return {"rewards": sum(self.rewards), "cumulative_rewards": self.cumulative_reward, "arm_counter": self.pulled_arm_counter}