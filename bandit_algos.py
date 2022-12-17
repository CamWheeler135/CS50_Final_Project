''' Classes of the different multi-armed bandit algorithms. '''

# Imports
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
        ''' For each time step, the agent will select and action. '''

        for i in range(self.number_of_pulls):
            selected_action = np.random.choice(self.env.num_of_bandits)
            reward, _, _, _ = self.env.step(selected_action)
            self.rewards.append(reward)
            self.cumulative_reward.append(sum(self.rewards) / len(self.rewards))
            self.pulled_arm_counter[selected_action] += 1
        
        return {"rewards": self.rewards, "cumulative_rewards": self.cumulative_reward, "arm_counter": self.pulled_arm_counter}





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


    def select_greedy_action(self, q_values):
        ''' Selects an bandit to pull, if q values for any bandits are equal
            function will break ties randomly. Returning the action to be selected. '''

        # Finds array of indices of possible actions (if there is a tie there will be more than 1 element in the array.)
        possible_actions = np.where(q_values == q_values[np.argmax(q_values)])

        # In the case of ties, choose on of those actions randomly.  
        selected_action = np.random.choice(possible_actions[0])
        return selected_action

    def update_Q(self, selected_action, reward, action_count):
        ''' Updates the Q value for the action by averaging the immediate rewards
            over the times that action has been selected. '''

        updated_value = self.Q_values[selected_action] + 1 / action_count[selected_action] * (reward - self.Q_values[selected_action])
        self.Q_values[selected_action] = updated_value

    def perform_actions(self):
        ''' For each time step the agent will select an action according
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


        return {"rewards": self.rewards, "cumulative_rewards": self.cumulative_reward, "arm_counter": self.pulled_arm_counter}





class EpsilonGreedyAgent():
    ''' Agent that takes the action with the highest Q value, but will continue to explore at probability epsilon. '''

    def __init__(self, env, number_of_pulls, epsilon=0.2):
        self.env = env
        self.number_of_pulls = number_of_pulls

        # States internal to the agent.
        self.rewards = []
        self.cumulative_reward = []
        self.pulled_arm_counter = np.zeros(self.env.num_of_bandits)
        self.Q_values = np.zeros(self.env.num_of_bandits)
        self.epsilon = epsilon

    
    def select_action(self, epsilon, q_values):
        ''' Chooses and action to pick, will choose highest Q value, breaking ties randomly. 
            Will choose a non-greedy action with probability epsilon. '''

        possible_actions = np.where(q_values == q_values[np.argmax(q_values)])

        # Choose the greedy action, breaking ties randomly.
        if np.random.random() > epsilon:
            # Select action from array of top actions, if top action = 1 then then that will just be selected.
            selected_action = np.random.choice(possible_actions[0])
            return selected_action
        # If value less than epsilon, choose a random bandit to pull.
        else:
            selected_action = np.random.choice(self.env.num_of_bandits)
            return selected_action

    def update_Q(self, selected_action, reward, action_count):
        ''' Updates the Q value for the action by averaging the immediate rewards
            over the times that action has been selected. '''

        updated_value = self.Q_values[selected_action] + 1 / action_count[selected_action] * (reward - self.Q_values[selected_action])
        self.Q_values[selected_action] = updated_value

    def perform_actions(self):
        ''' For each time step the agent will select an action according
            according to the highest action value (q_value) but select a random
            action according to probability epsilon '''

        for i in range(self.number_of_pulls):

            # We need to tell the agent how to make decisions if action values are equal.
            selected_action = self.select_action(q_values=self.Q_values, epsilon=self.epsilon)
            reward, _, _, _ = self.env.step(selected_action)
            self.rewards.append(reward)
            self.cumulative_reward.append(sum(self.rewards) / len(self.rewards))
            self.pulled_arm_counter[selected_action] += 1

            # Update the Q value of that action.
            self.update_Q(selected_action=selected_action, reward=reward, action_count=self.pulled_arm_counter)


        return {"rewards": self.rewards, "cumulative_rewards": self.cumulative_reward, "arm_counter": self.pulled_arm_counter}





class UCBAgent():
    ''' Agent uses the upper confidence bounds to compute what actions to select. This agent uses the UCB1 algorithm. '''

    def __init__(self, env, number_of_pulls) -> None:
        self.env = env
        self.number_of_pulls = number_of_pulls

        # States internal of the agent. 
        self.rewards = []
        self.cumulative_reward = []
        self.pulled_arm_counter = np.zeros(self.env.num_of_bandits)
        # Values used in the UCB computation.
        self.Q_values = np.zeros(self.env.num_of_bandits)
        self.Ut_values = np.zeros(self.env.num_of_bandits)
        self.UCB = np.zeros(self.env.num_of_bandits)
        self.time_steps = 0
        # Ensures all actions are selected once before using UCB.
        self.first_action_counter = 0 

    def update_confidence(self, selected_action, action_count, time_steps):
        ''' Computes and updates the confidence bound for each bandit. '''

        updated_confidence = np.sqrt((2 * np.log(time_steps)/ action_count[selected_action]))
        self.Ut_values[selected_action] = updated_confidence

    def update_Q(self, selected_action, reward, action_count):
        ''' Updates the Q value for the action by averaging the immediate rewards
            over the times that action has been selected. '''

        updated_value = self.Q_values[selected_action] + 1 / action_count[selected_action] * (reward - self.Q_values[selected_action])
        self.Q_values[selected_action] = updated_value

    def select_action(self, first_action_counter):
        ''' Selections actions by choosing the value that maximizes Qt + Ut. '''

        # Choose each action once, then start using the UCB algorithm.
        while first_action_counter < self.env.num_of_bandits:
            return first_action_counter

        # Compute the action for selection.
        for bandit in range(self.env.num_of_bandits):
            self.UCB[bandit] = self.Q_values[bandit] + self.Ut_values[bandit]
        selected_action = np.argmax(self.UCB)
        return selected_action

    def perform_actions(self):
        ''' For each time step the agent will select an action one time
            before changing to the UCB algorithm. '''

        for i in range(self.number_of_pulls):

            selected_action = self.select_action(first_action_counter=self.first_action_counter)
            reward, _, _, _ = self.env.step(selected_action)
            self.rewards.append(reward)
            self.cumulative_reward.append(sum(self.rewards) / len(self.rewards))
            self.pulled_arm_counter[selected_action] += 1

            # Update Q and Ut values
            self.time_steps += 1
            self.first_action_counter += 1
            self.update_confidence(selected_action=selected_action, action_count=self.pulled_arm_counter, time_steps=self.time_steps)
            self.update_Q(selected_action=selected_action, reward=reward, action_count=self.pulled_arm_counter)
        
        return {"rewards": self.rewards, "cumulative_rewards": self.cumulative_reward, "arm_counter": self.pulled_arm_counter}