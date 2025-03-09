import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def softmax(x):
    exp_x = np.exp(x - np.max(x)) 
    return exp_x / np.sum(exp_x)


def ETCRegretBound(explore_steps, num_arms, num_timesteps, max_mean, means):
    means = np.asarray(means)  # Ensure means is an array
    RegretUpperBound = (
        explore_steps * np.sum(max_mean - means) +
        (num_timesteps - num_arms * explore_steps) *
        np.sum((max_mean - means) * np.exp(- (explore_steps * (max_mean - means) ** 2) / 4))
    )
    return RegretUpperBound


class Bandit:
    def __init__(self, num_arms, distribution='gaussian', means=None, random_mode=False, reward_gap=None):
        self.num_arms = num_arms
        self.distribution = distribution.lower()
        self.reward_gap = reward_gap
        
        if means is not None:
            self.means = np.array(means)
        else:
            if self.distribution == 'gaussian':
                self.means = np.random.standard_normal(num_arms)
            elif self.distribution == 'bernoulli':
                self.means = np.random.uniform(0, 1, num_arms)
            else:
                raise ValueError("Unsupported distribution type. Choose 'gaussian' or 'bernoulli'.")
        
        max_mean = np.max(self.means)
        
        if random_mode and reward_gap is not None:
            self._apply_reward_gap(max_mean)

    def _apply_reward_gap(self, max_mean):
        sorted_means = [max_mean - k * self.reward_gap for k in range(self.num_arms)]
        if self.distribution == 'bernoulli':
            sorted_means = [max(0, m) for m in sorted_means]
        self.means = np.array(sorted_means)

    def pull_arm(self, arm):
        if arm < 0 or arm >= self.num_arms:
            raise ValueError("Invalid arm index")
        if self.distribution == 'gaussian':
            return np.random.normal(self.means[arm], 1)
        elif self.distribution == 'bernoulli':
            return np.random.binomial(1, self.means[arm])
        else:
            raise ValueError("Unsupported distribution type.")
    
    def __repr__(self):
        return np.array(self.means)
    
    def optimize(self, num_timesteps):
        result = minimize_scalar(
                ETCRegretBound,
                bounds=(1, num_timesteps // self.num_arms),
                args=(self.num_arms, num_timesteps, np.max(self.means), self.means),
                method='bounded'
            )
        return max(1, int(result.x) + 1)




class Game:
    def __init__(self, num_arms, distribution, means, random_mode, reward_gap, explore_steps, num_timesteps, Iterations ,AlgorithmDependendVariable):
        self.num_arms = num_arms
        self.distribution = distribution
        self.means = means
        self.random_mode = random_mode
        self.reward_gap = reward_gap
        self.explore_steps = explore_steps
        self.num_timesteps = num_timesteps
        self.Iterations = Iterations
        self.AlgorithmDependendVariable = AlgorithmDependendVariable
        
        
    
    
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
            #Decide what learning algorithm to use
            
            action = agent.select_action(np.random.uniform(0, 1))
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

    


    def AverageGame(self, bandit, max_mean, means, agent):
        data = {
            'time': np.arange(self.num_timesteps),
            'regret': np.zeros(self.num_timesteps),
            'correct_action_rate': np.zeros(self.num_timesteps),
            'estimated_means': [np.zeros(self.num_arms) for _ in range(self.num_timesteps)],
            'probabilities': [np.zeros(self.num_arms) for _ in range(self.num_timesteps)],
            'regret_var': np.zeros(self.num_timesteps),
            'correct_action_rate_var': np.zeros(self.num_timesteps),
            'estimated_means_var': [np.zeros(self.num_arms) for _ in range(self.num_timesteps)],
            'probabilities_var': [np.zeros(self.num_arms) for _ in range(self.num_timesteps)],
            'regret_CI_upper': np.zeros(self.num_timesteps)
        }
        df = pd.DataFrame(data)

        
        dataForVar = [None] * self.num_timesteps  # Creates a list that can hold DataFrames
        CompleteDataForVar = [None] * self.Iterations
        
        for i in range(self.Iterations):
            dfSingle = self.GameOnce(self.num_arms, means, self.num_timesteps, bandit, max_mean, agent)
            for t in range(self.num_timesteps):
                if i == 0:
                    df.at[t, 'regret'] = dfSingle.at[t, 'regret']
                    df.at[t, 'correct_action_rate'] = dfSingle.at[t, 'correct_action_rate']
                    df.at[t, 'probabilities'] = dfSingle.at[t, 'probabilities']
                    df.at[t, 'estimated_means'] = dfSingle.at[t, 'estimated_means']
                else:
                    prev_mean_regret = df.at[t, 'regret']
                    prev_mean_correct_action_rate = df.at[t, 'correct_action_rate']
                    prev_mean_estimated_means = df.at[t, 'estimated_means']
                    prev_mean_probabilities = df.at[t, 'probabilities']
                    
                    df.at[t, 'regret'] += (prev_mean_regret *(-1) + dfSingle.at[t, 'regret']) / (i + 1)
                    df.at[t, 'correct_action_rate'] += (prev_mean_correct_action_rate *(-1) + dfSingle.at[t, 'correct_action_rate']) / (i + 1)
                    df.at[t, 'probabilities'] += (prev_mean_probabilities  *(-1)+ dfSingle.at[t, 'probabilities']) / (i + 1)
                    df.at[t, 'estimated_means'] += (prev_mean_estimated_means *(-1) + dfSingle.at[t, 'estimated_means']) / (i + 1)
            
                dataForVar[t] = dfSingle     
            CompleteDataForVar[i] = list(dataForVar)

        for t in range(self.num_timesteps):  
            regret_samples = [CompleteDataForVar[i][t].at[t, 'regret'] for i in range(self.Iterations)]
            correct_action_samples = [CompleteDataForVar[i][t].at[t, 'correct_action_rate'] for i in range(self.Iterations)]

            df.at[t, 'regret_var'] = np.var(regret_samples, ddof=1)
            df.at[t, 'correct_action_rate_var'] = np.var(correct_action_samples, ddof=1)      
            df.at[t, 'estimated_means_var'] = np.var([CompleteDataForVar[i][t].at[t, 'estimated_means'] for i in range(self.Iterations)], axis=0)
            df.at[t, 'probabilities_var'] = np.var([CompleteDataForVar[i][t].at[t, 'probabilities'] for i in range(self.Iterations)], axis=0)
            
            # Confidence Interval Calculation
            regret_std = np.sqrt(df.at[t, 'regret_var'])
            correct_action_std = np.sqrt(df.at[t, 'correct_action_rate_var'])

            ci_factor = 1.96 / np.sqrt(self.Iterations)  # 95% CI (Normal Approximation)

            df.at[t, 'regret_CI_upper'] = df.at[t, 'regret'] + ci_factor * regret_std

        
        return df

    

        




class ExploreThenCommit:
    def __init__(self, n_actions, explore_steps):
        self.n_actions = n_actions
        self.explore_steps = explore_steps
        self.q_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)
        self.step = 0

    def select_action(self,SomeVariableWeDoNotNeed):
        if self.step < self.explore_steps:
            return np.random.choice(self.n_actions)
        return np.argmax(self.q_values)

    def update_estimates(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]
        self.step += 1


class EpsilonGreedy:

    def __init__(self, n_actions, epsilon):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.q_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)
        self.step = 0
        

    def select_action(self,random):
        if random < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_values)
        
    def update_estimates(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]
        self.step += 1
        
class Greedy:
    def __init__(self, n_actions,SomethingWeDoNotNeed):
        self.n_actions = n_actions
        self.q_values = np.ones(n_actions)*1000000
        self.action_counts = np.zeros(n_actions)
        self.step = 0

    def select_action(self,SomeVariableWeDoNotNeed):
        return np.argmax(self.q_values)
    
    def update_estimates(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]
        self.step += 1


class UCB:
    def __init__(self, n_actions,delta):
        self.n_actions = n_actions
        self.q_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)
        self.step = 0
        self.delta = delta

        
    def select_action(self,SomeVariableWeDoNotNeed):
        if 0 in self.action_counts:
            return np.argmin(self.action_counts)
        ucb_values = self.q_values + np.sqrt( (2 * np.log(1/self.delta)) / self.action_counts) 
        ArmChosen=np.argmax(ucb_values)
        self.action_counts[ArmChosen] += 1
        self.step += 1
        return ArmChosen
    
    def update_estimates(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]
        self.step += 1


class Boltzmann:
    def __init__(self, n_actions,theta):
        self.n_actions = n_actions
        self.q_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)
        self.step = 0
        self.theta = theta

    def select_action(self,SomeVariableWeDoNotNeed):
        probabilities = np.exp(self.q_values / self.theta) / np.sum(np.exp(self.q_values / self.theta))
        return np.random.choice(self.n_actions, p=probabilities)
    
    def update_estimates(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]
        self.step += 1


class PolicyGradient:
    def __init__(self, n_actions,alphaBaselineThetaList):
        self.n_actions = n_actions
        self.q_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)
        self.step = 0
        self.alpha = alphaBaselineThetaList[0]
        self.decayRateOfStepSize = alphaBaselineThetaList[1]
        self.baseline = alphaBaselineThetaList[2]
        self.theta = alphaBaselineThetaList[3]
        self.probabilities = np.exp(self.q_values / self.theta) / np.sum(np.exp(self.q_values / self.theta))

    def select_action(self,SomeVariableWeDoNotNeed):
        
        return np.random.choice(self.n_actions, p=self.probabilities)

    def update_estimates(self, action, reward):
        self.probabilities = np.exp(self.q_values / self.theta) / np.sum(np.exp(self.q_values / self.theta))
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action] 
        grad = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            if a==action:
                self.theta[a] += self.alpha * (reward - self.q_values[a]) * (1 - self.probabilities[action])
            else:
                self.theta[a] -= self.alpha * (reward - self.q_values[a]) *(self.probabilities[action])
            
        self.alpha *= self.decayRateOfStepSize


class EpsilonDecreasing:

    def __init__(self, n_actions, EpsilonAndDecayRateList):
        self.n_actions = n_actions
        self.epsilon = EpsilonAndDecayRateList[0]
        self.decayRate = EpsilonAndDecayRateList[1]
        self.q_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)
        self.step = 0
        

    def select_action(self,random):
        if random < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_values)
        
    def update_estimates(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]
        self.step += 1
        self.epsilon *= self.decayRate

    

