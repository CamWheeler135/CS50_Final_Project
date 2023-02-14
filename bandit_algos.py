''' Classes of the different multi-armed bandit algorithms. '''

# Imports
import numpy as np
from tqdm import tqdm


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
        self.regret = []
        self.cumulative_regret = []
        self.pulled_arm_counter = np.zeros(self.env.num_of_bandits)

    def perform_actions(self):
        ''' For each time step, the agent will select and action. '''

        max_reward = self.env.rewards[np.argmax(self.env.rewards)]
        print(max_reward)
        for i in tqdm(range(self.number_of_pulls)):
            # Select an action, receive a reward, compute cumulative reward and regret and increment arm counter
            selected_action = np.random.choice(self.env.num_of_bandits)
            reward, _, _, _ = self.env.step(selected_action)
            self.rewards.append(reward)
            self.cumulative_reward.append(sum(self.rewards) / len(self.rewards))
            action_regret = np.max(self.env.reward_probas) - self.env.reward_probas[selected_action]
            self.regret.append(action_regret)
            self.cumulative_regret.append(sum(self.regret))
            self.pulled_arm_counter[selected_action] += 1
        
        return {"rewards": self.rewards, "cumulative_rewards": self.cumulative_reward, "arm_counter": self.pulled_arm_counter, "regret": self.cumulative_regret}


class GreedyAgent():
    ''' An agent that selects the action with the highest
        action value 100% of the time. '''

    def __init__(self, env, number_of_pulls): 
        self.env = env
        self.number_of_pulls = number_of_pulls
        
        # Internal states of the agent, note that we also have to add the notion of an "action value" (q_value)
        self.rewards = []
        self.cumulative_reward = []
        self.regret = []
        self.cumulative_regret = []
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

        max_reward = self.env.rewards[np.argmax(self.env.rewards)]

        for i in tqdm(range(self.number_of_pulls)):

            # Select an action greedily, get a reward, compute cumulative reward and regret and increment arm counter and update Q. 
            selected_action = self.select_greedy_action(q_values=self.Q_values)
            reward, _, _, _ = self.env.step(selected_action)
            self.rewards.append(reward)
            self.cumulative_reward.append(sum(self.rewards) / len(self.rewards))
            action_regret = np.max(self.env.reward_probas) - self.env.reward_probas[selected_action]
            self.regret.append(action_regret)
            self.cumulative_regret.append(sum(self.regret))
            self.pulled_arm_counter[selected_action] += 1

            # Update the Q value of that action.
            self.update_Q(selected_action=selected_action, reward=reward, action_count=self.pulled_arm_counter)


        return {"rewards": self.rewards, "cumulative_rewards": self.cumulative_reward, "arm_counter": self.pulled_arm_counter, 
                "q_values": self.Q_values, "regret": self.cumulative_regret}


class EpsilonGreedyAgent():
    ''' Agent that takes the action with the highest Q value, but will continue to explore at probability epsilon. '''

    def __init__(self, env, number_of_pulls, epsilon=0.2):
        self.env = env
        self.number_of_pulls = number_of_pulls

        # States internal to the agent.
        self.rewards = []
        self.cumulative_reward = []
        self.regret = []
        self.cumulative_regret = []
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

    def update_Q(self, selected_action, reward, action_count: np.array):
        ''' Updates the Q value for the action by averaging the immediate rewards
            over the times that action has been selected. '''

        updated_value = self.Q_values[selected_action] + 1 / action_count[selected_action] * (reward - self.Q_values[selected_action])
        self.Q_values[selected_action] = updated_value

    def perform_actions(self):
        ''' For each time step the agent will select an action according
            according to the highest action value (q_value) but select a random
            action according to probability epsilon '''

        max_reward = self.env.rewards[np.argmax(self.env.rewards)]

        for i in tqdm(range(self.number_of_pulls)):

            # Select an action, receive reward, compute cumulative reward and regret, increment arm counter and update Q.
            selected_action = self.select_action(q_values=self.Q_values, epsilon=self.epsilon)
            reward, _, _, _ = self.env.step(selected_action)
            self.rewards.append(reward)
            self.cumulative_reward.append(sum(self.rewards) / len(self.rewards))
            action_regret = np.max(self.env.reward_probas) - self.env.reward_probas[selected_action]
            self.regret.append(action_regret)
            self.cumulative_regret.append(sum(self.regret))
            self.pulled_arm_counter[selected_action] += 1

            # Update the Q value of that action.
            self.update_Q(selected_action=selected_action, reward=reward, action_count=self.pulled_arm_counter)


        return {"rewards": self.rewards, "cumulative_rewards": self.cumulative_reward, "arm_counter": self.pulled_arm_counter,
                "q_values": self.Q_values, "regret": self.cumulative_regret}


class EGDecayAgent(EpsilonGreedyAgent):
    ''' Agent that takes the action with the highest Q values, but will explore other actions with probability epsilon.
        This implementation will make epsilon dynamic, causing it to decay over time. Meaning towards the end, the agent
        should exploit more. '''

    def __init__(self, env, number_of_pulls, epsilon=1, epsilon_decay_rate=0.99, epsilon_min = 0.1):
        super().__init__(env, number_of_pulls, epsilon)
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_values = []


    def decay_epsilon(self):
        ''' This function will decay epsilon in order to decrease exploration as time steps increase. '''
        decayed_epsilon = self.epsilon * self.epsilon_decay_rate

        return decayed_epsilon

    def perform_actions(self):

        max_reward = self.env.rewards[np.argmax(self.env.rewards)]
        
        for i in tqdm(range(self.number_of_pulls)):

        # Select an action, receive reward, compute cumulative reward and regret, increment arm counter and update Q.
            selected_action = self.select_action(q_values=self.Q_values, epsilon=self.epsilon)
            reward, _, _, _ = self.env.step(selected_action)
            self.rewards.append(reward)
            self.cumulative_reward.append(sum(self.rewards) / len(self.rewards))
            action_regret = np.max(self.env.reward_probas) - self.env.reward_probas[selected_action]
            self.regret.append(action_regret)
            self.cumulative_regret.append(sum(self.regret))
            self.pulled_arm_counter[selected_action] += 1

            # Update the Q value of that action and decay epsilon if epsilon > min epsilon value.
            self.update_Q(selected_action=selected_action, reward=reward, action_count=self.pulled_arm_counter)
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.decay_epsilon()
            self.epsilon_values.append(self.epsilon)

        return {"rewards": self.rewards, "cumulative_rewards": self.cumulative_reward, "arm_counter": self.pulled_arm_counter,
                "q_values": self.Q_values, "regret": self.cumulative_regret, 'decay': self.epsilon_values}


class UCBAgent():
    ''' Agent uses the upper confidence bounds to compute what actions to select. This agent uses the UCB1 algorithm. '''

    def __init__(self, env, number_of_pulls, c=2):
        self.env = env
        self.number_of_pulls = number_of_pulls

        # States internal of the agent. 
        self.rewards = []
        self.cumulative_reward = []
        self.regret = []
        self.cumulative_regret = []
        self.pulled_arm_counter = np.zeros(self.env.num_of_bandits)

        # Values used in the UCB computation.
        self.Q_values = np.zeros(self.env.num_of_bandits)
        self.Ut_values = np.zeros(self.env.num_of_bandits)
        self.UCB = np.zeros(self.env.num_of_bandits)
        self.c = c
        self.time_steps = 0

    def update_confidence(self, action_count, time_steps):
        ''' Computes and updates the confidence bound for each bandit. '''

        # Loops through each of the confidence bounds and updated them accordingly. 
        for i in range(np.size(self.Ut_values)):
            updated_confidence = self.c * np.sqrt(np.log(time_steps) / action_count[i])
            self.Ut_values[i] = updated_confidence

    def update_Q(self, selected_action, reward, action_count):
        ''' Updates the Q value for the action by averaging the immediate rewards
            over the times that action has been selected. '''

        updated_value = self.Q_values[selected_action] + 1 / action_count[selected_action] * (reward - self.Q_values[selected_action])
        self.Q_values[selected_action] = updated_value

    def select_action(self, q_values: np.array, ut_values: np.array, ucb: np.array):
        ''' Selections actions by choosing the value that maximizes Qt + Ut. '''

        # Choose each action once, then start using the UCB algorithm.
        while self.time_steps < self.env.num_of_bandits:
            return self.time_steps 

        # Compute the action for selection.
        for bandit in range(self.env.num_of_bandits):
            ucb[bandit] = q_values[bandit] + ut_values[bandit]
        selected_action = np.argmax(ucb)
        return selected_action

    def perform_actions(self):
        ''' Agent will select an action one time
            before changing to the UCB algorithm. '''
        
        max_reward = self.env.rewards[np.argmax(self.env.rewards)]

        for i in tqdm(range(self.number_of_pulls)):

            selected_action = self.select_action(q_values=self.Q_values, ut_values=self.Ut_values, ucb=self.UCB)
            reward, _, _, _ = self.env.step(selected_action)
            self.rewards.append(reward)
            self.cumulative_reward.append(sum(self.rewards) / len(self.rewards))
            action_regret = np.max(self.env.reward_probas) - self.env.reward_probas[selected_action]
            self.regret.append(action_regret)
            self.cumulative_regret.append(sum(self.regret))
            self.pulled_arm_counter[selected_action] += 1

            # Check that all actions have been pulled at least once before using UCB.
            if self.time_steps < self.env.num_of_bandits:
                self.update_Q(selected_action=selected_action, reward=reward, action_count=self.pulled_arm_counter)
                self.time_steps += 1
            else:
                # Update Q and Ut values
                self.update_Q(selected_action=selected_action, reward=reward, action_count=self.pulled_arm_counter)
                self.update_confidence(action_count=self.pulled_arm_counter, time_steps=self.time_steps)
                self.time_steps +=1
        
        return {"rewards": self.rewards, "cumulative_rewards": self.cumulative_reward, "arm_counter": self.pulled_arm_counter, 
                "q_values": self.Q_values, "confidence": self.Ut_values, "regret": self.cumulative_regret}


class ThompsonSamplingAgent():
    '''' Agent that performs Thompson Sampling. '''

    def __init__(self, env, number_of_pulls):
        self.env = env
        self.number_of_pulls = number_of_pulls

        # States internal of the agent. 
        self.rewards = []
        self.cumulative_reward = []
        self.regret = []
        self.cumulative_regret = []
        self.pulled_arm_counter = np.zeros(self.env.num_of_bandits)

        # Values needed for Thompson Sampling
        # Prior distributions, assuming uniform distribution between 0 and 1 (we know very little).
        self.prior_dists = np.asarray([np.random.uniform(low=0, high=1) for i in range(self.env.num_of_bandits)])
        # Posterior beta distribution, first array = a, second array = b.
        self.posterior_dists = np.ones([2, self.env.num_of_bandits])
        self.action_counter = 0

    def select_action(self, prior_dists, posterior_dists, action_counter):
        ''' Selects and action according to the Thompson Sampling Algorithm. '''
        
        if action_counter == 0:
            return np.argmax(prior_dists)
        else:
            # Samples the beta distribution with the a and b values contained in the posterior distribution array.
            posterior_samples = np.asarray([np.random.beta(a=posterior_dists[0, i], b=posterior_dists[1, i]) for i in range(self.env.num_of_bandits)])
            return np.argmax(posterior_samples)

    def update_posterior(self, selected_action, reward, posterior_dists):
        ''' Updated the posterior beta distribution according to the reward we received from taking that action. '''

        # If the reward equals 1 we increment a, if the reward equals 0 we increment b (these are parameters of the beta distribution).
        if reward == 1:
            posterior_dists[0, selected_action] += 1
        else:
            posterior_dists[1, selected_action] += 1
    
    
    def perform_actions(self):
        ''' For each time step the agent will select and action computed through thompson sampling 
            and perform that action. '''
        
        max_reward = self.env.rewards[np.argmax(self.env.rewards)]

        for i in tqdm(range(self.number_of_pulls)):
            selected_action = self.select_action(prior_dists=self.prior_dists, posterior_dists=self.posterior_dists, action_counter=self.action_counter)
            reward, _, _, _ = self.env.step(selected_action)

            # Compute metrics.
            self.rewards.append(reward)
            self.cumulative_reward.append(sum(self.rewards) / len(self.rewards))
            action_regret = np.max(self.env.reward_probas) - self.env.reward_probas[selected_action]
            self.regret.append(action_regret)
            self.cumulative_regret.append(sum(self.regret))
            self.pulled_arm_counter[selected_action] += 1
            self.action_counter += 1

            # Update posterior.
            self.update_posterior(selected_action=selected_action, reward=reward, posterior_dists=self.posterior_dists)
        
        return {"rewards": self.rewards, "cumulative_rewards": self.cumulative_reward, "arm_counter": self.pulled_arm_counter, 
                "posterior_dists": self.posterior_dists, "regret": self.cumulative_regret}