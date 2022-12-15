
# Welcome to my CS50 final project. 

### 1.0 Introduction.

Here I will be exploring reinforcement learning (RL), a type of machine learning that aims to interact with an environment in order to improve. Unlike machine learning methods you are probably used to (do not worry if you are not) RL needs to interact with this environment in order to learn. A great example of RL comes from DeepMind where they produced an agent that learnt to play atari games to a human level and above! This project will not be tackling tasks at that level however, I will be exploring what is called "The multi-armed bandit problem". This is a much simpler version of the full RL problem and can be thought of like going to the casino and playing the slot machines. I will explain it in much more detail below. 

Once this project is complete, feel free to clone the repo and have a play around with the code, I will also include my references and links to their blogs/code at the bottom of the file. Jump straight there to see my inspiration. 

---

### 2.0 Multi Armed Bandit Problem

You are at a casino, there are plenty of slot machines around you. these are the one armed bandits, (named according because slot machines used to have a big leaver to pull). Each of these bandits return to us some reward (this could be positive or negative), but we do not know what that reward is. What is the best strategy to obtain the most profit? Which bandits should we pull the most so we leave with the most money? Or do we just let luck do its thing and hope we leave with more money than we entered with? 

This problem is also called the "explore or exploit" problem. The idea of when should we explore for more knowledge (try out that bandit we haven't pulled yet to see if it gives us more profit), or exploit the knowledge we have already (stick on the bandit we know is currently giving us the most profit). Researchers have been looking at this problem and have implemented some very cool algorithms to try and solve this problem! There are also several evolutions to this problem, such as what if the reward for each bandit changes with each pull? 


### 2.1. The Environment. 

We need the environment to do several things for this problem. 1st, we need to set the number of bandits we need, then mean return of each bandit and the probability of us getting that reward. For now we are going to create a stationary bandit, where the reward remains the same through the process. However, later on, I will implement a bandit where the returns are dynamic, sampled from a reward distribution. This should make the problem slightly harder for the algorithms to learn from.

For this set up, we are going to create and environment class, where we can create instances of the environment and pass them in as parameters when we are creating our agents. 

The environment must do certain things.
- Take in an array or rewards and reward probabilites as arguments.
- Given an action, return the reward to the agent. 

To create the environment, I will be using [OpenAI's Gym](https://www.gymlibrary.dev).

### 2.2. Agents

There are plenty of algorithms that have been created to solve bandit problems (Do not worry if some of them look scary, I will try my best to explain how each of them works). In this implementation of the problem, our agent will take in the environment as a parameter and conduct its actions. The actions thea agent takes and rewards of its actions will be stored in the agent (we need this compute several things when we get to the more difficult algorithms) and our agent should present these results to a few plotting functions that will create some nice shiny graphs for us.

The agents I plan to implement are:
    1. Random Agent.
    2. Greedy Agent.
    3. $\epsilon$-greedy Agent.
    4. Upper Confidence Bound (UCB) Agent.
    5. Thompson Sampling Agent. 

#### The Random Agent.

This agent has no concept of what is going on really, they simply just pull whatever bandit they want, regardless of reward, or how many times they have pulled it. This idea can be thought of as simply closing your eyes at the casino and pulling any of the slot machines you fancy.

---

### References

- [DeepMind x UCL Lectures](https://youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm) - Lectures 1 and 2 helped my mathematical and theoretical understanding of RL, multi-armed bandits and the algorithms. 

- [Alejandro Aristzabal](https://medium.com/@alejandro.aristizabal24) - Has a great series on the multi-armed bandit problem, with code. 

- [Edward Pie](https://www.youtube.com/watch?v=sNamSTJ4qCU) - Great explanation with some example code.

