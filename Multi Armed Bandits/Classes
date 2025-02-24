import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / np.sum(exp_x)


def OptimizeExploration(explore_steps, num_arms, num_timesteps, max_mean, means):
    means = np.asarray(means)  # Ensure means is an array
    RegretUpperBound = (
        explore_steps * np.sum(max_mean - means) +
        (num_timesteps - num_arms * explore_steps) *
        np.sum((max_mean - means) * np.exp(- (explore_steps * (max_mean - means) ** 2) / 4))
    )
    return RegretUpperBound

class Bandit:
    def __init__(self, num_arms, distribution='gaussian', means=None, random_mode=False, reward_gap=None):
        self.num_arms = num_arms #number of arms
        self.distribution = distribution.lower() #writes a string in lowercase
        self.reward_gap = reward_gap 
        
        if means is not None: #Initializes the means of the arms in the following three cases
            self.means = np.array(means) #If the means are given as a List
        else:
            if self.distribution == 'gaussian': #Initializes the means in the Gaussian case
                self.means = np.random.standard_normal(num_arms)
            elif self.distribution == 'bernoulli': #Initializes the means in the Bernoulli case
                self.means = np.random.uniform(0, 1, num_arms)
            else:
                raise ValueError("Unsupported distribution type. Choose 'gaussian' or 'bernoulli'.") #Mispelling error
            
        max_mean = np.max(self.means) #Highest mean

        if random_mode and reward_gap is not None: #Initialize the means over a specified reward gap
            self._apply_reward_gap(max_mean)
        
    
    def _apply_reward_gap(self,max_mean): #This function does not have to be called in the file Game, because then we could get an error, in the case that reward_gap=None
        sorted_means = [max_mean - k * self.reward_gap for k in range(self.num_arms)]
        if self.distribution == 'bernoulli': #In Bernulli case we need to set the negative means to zero
            sorted_means = [max(0, m) for m in sorted_means]  # Ensure non-negative means
        self.means = np.array(sorted_means)
    
    def pull_arm(self, arm):
        if arm < 0 or arm >= self.num_arms:
            raise ValueError("Invalid arm index")
        
        if self.distribution == 'gaussian':
            return np.random.normal(self.means[arm], 1)  # Standard deviation is for all arms equal to 1
        elif self.distribution == 'bernoulli':
            return np.random.binomial(1, self.means[arm])
        else:
            raise ValueError("Unsupported distribution type.")
    
    def __repr__(self):
        return np.array(self.means)
    
    def optimize(self,ArbitraryExplorationSteps,max_mean,num_timesteps):
        if ArbitraryExplorationSteps is None:
            result = minimize_scalar(
                OptimizeExploration,
                bounds=(1, num_timesteps // self.num_arms),
                args=(self.num_arms, num_timesteps, max_mean, self.means),  # Directly pass means as a NumPy array
                method='bounded'
            )

            explore_steps = np.max([1,int(result.x)+1])
            print("Optimal Exploration Steps: ", explore_steps)
        else:
            explore_steps=ArbitraryExplorationSteps
            print("The arbitrary Exploration Steps: ", explore_steps)
        return explore_steps
        




class Game:
    def __init__(self, num_arms, distribution, means, random_mode, reward_gap, explore_steps, num_timesteps, Iterations):
        self.num_arms = num_arms
        self.distribution = distribution
        self.means = means
        self.random_mode = random_mode
        self.reward_gap = reward_gap
        self.explore_steps = explore_steps
        self.num_timesteps = num_timesteps
        self.Iterations = Iterations
        
        
    
    
    def GameOnce(self, num_arms, means, num_timesteps, bandit, max_mean, agent):
        
        dataSingle = {
            'time': np.arange(num_timesteps),
            'regret': np.zeros(num_timesteps),
            'correct_action_rate': np.zeros(num_timesteps),
            'estimated_means': [np.zeros(num_arms) for _ in range(num_timesteps)],  # Store as lists of arrays
            'probabilities': [np.zeros(num_arms) for _ in range(num_timesteps)]  # Store as lists of arrays
        }
        dfSingle = pd.DataFrame(dataSingle)
        for t in range(num_timesteps):
            action = agent.select_action()
            reward = bandit.pull_arm(action)
            agent.update_estimates(action, reward)
            #Calcluation of the regret over time and Calcluation of the correct action rate over time
            if t ==0:
                dfSingle.loc[t, 'regret'] = max_mean - means[action]
                dfSingle.loc[t, 'correct_action_rate'] = 1 if action == np.argmax(means) else 0
            else:
                dfSingle.loc[t, 'regret'] = dfSingle.loc[t-1, 'regret'] + (max_mean - means[action])
                dfSingle.loc[t, 'correct_action_rate'] = (dfSingle.loc[t-1, 'correct_action_rate'] * t + (1 if action == np.argmax(means) else 0)) / (t+1)
            dfSingle.at[t, 'probabilities'] = softmax(agent.q_values).copy()
            dfSingle.at[t, 'estimated_means'] = agent.q_values
        return dfSingle

    def AverageGame(self,bandit, max_mean, means, agent):
        
        data = {
            'time': np.arange(self.num_timesteps),
            'regret': np.zeros(self.num_timesteps),
            'correct_action_rate': np.zeros(self.num_timesteps),
            'estimated_means': [np.zeros(self.num_arms) for _ in range(self.num_timesteps)],  # Store as lists of arrays
            'probabilities': [np.zeros(self.num_arms) for _ in range(self.num_timesteps)]  # Store as lists of arrays
        }
        df = pd.DataFrame(data)
        for i in range(self.Iterations):
            dfSingle=self.GameOnce(self.num_arms, means, self.num_timesteps, bandit, max_mean, agent)
            if i == 0:
                for t in range(self.num_timesteps):
                    df.at[t, 'regret'] = dfSingle.at[ t, 'regret']
                    df.at[t, 'correct_action_rate'] = dfSingle.at[t, 'correct_action_rate']
                    df.at[ t, 'probabilities']=dfSingle.at[t, 'probabilities']
                    df.at[ t, 'estimated_means'] = dfSingle.at[t, 'estimated_means']
            else:
                for t in range(self.num_timesteps):
                    df.at[t, 'regret'] = (df.at[t, 'regret'] * i + dfSingle.at[ t,'regret']) / (i+1)
                    df.at[t, 'correct_action_rate'] = (df.at[t, 'correct_action_rate'] * i + dfSingle.at[t, 'correct_action_rate']) / (i+1)
                    df.at[t, 'probabilities'] = (df.at[t, 'probabilities'] * i + dfSingle.at[t, 'probabilities']) / (i+1)
                    df.at[t, 'estimated_means'] = (df.at[t, 'estimated_means'] * i + dfSingle.at[t, 'estimated_means']) / (i+1)
        return df

    

class ExploreThenCommit:
    def __init__(self, n_actions, explore_steps):
        self.n_actions = n_actions              # Number of possible actions
        self.explore_steps = explore_steps      # Number of exploration steps
        self.q_values = np.zeros(n_actions)     # Estimated action values
        self.action_counts = np.zeros(n_actions) # Counts of each action
        self.step = 0                            # Current step in the process

    def select_action(self):
        # Exploration phase: select actions uniformly
        if self.step < self.explore_steps:
            return np.random.choice(self.n_actions)
        # Exploitation phase: select the best action
        return np.argmax(self.q_values)

    def update_estimates(self, action, reward):
        # Update action counts and estimated values
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]
        self.step += 1
        


