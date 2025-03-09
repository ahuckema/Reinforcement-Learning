from Classes import Bandit, Game, ExploreThenCommit, EpsilonGreedy, UCB, Greedy, EpsilonDecreasing, PolicyGradient, Boltzmann
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar





reward_gap=None 


######### BEGIN INPUT BY THE USER ##########################################################################################

#Number of Arms and distribution: either bernoulli or gaussian
means=[1,2,3,4,6,7,10,11,11.5,12] #Either None or a list of means for the arms. Check that lenght of the list is equal to num_arms
random_mode=True #Either True or False. If False, then means are normally distributed around 0 with a standard deviation of 1, but made positive if negativly genereated.
#Only means=None and random_mode=False or means=[...] and random_mode=True are compatible
num_arms=10
distribution='gaussian'

#Number of Time Steps and how often to average the results
num_timesteps = 1000
Iterations=1000

#Choose the Agent
AgentList = ['ExploreThenCommit', 'EpsilonGreedy', 'UCB', 'Greedy', 'EpsilonDecreasing', 'PolicyGradient','Boltzmann']
ChosenAgent = 'PolicyGradient' #Choose from AgentList

#Choose the AlgorithmDependendVariable, i.e. epsilon for EpsilonGreedy, delta for UCB, Step size for PolicyGradient, None for Greedy, the exploration steps for ExploreThenCommit
#Policy Gradient: [step_size, DecayRateOfStepSize ,baseline = 'yes' or 'no' ,ThetaListOfNum_ArmsDimensions]
#EpsilonGreedy: epsilon
#UCB: delta
#Boltzmann: Theta
#ExploreThenCommit: None (for automatic optimization) or a number for a fixed number of exploration steps
#EpsilonDecreasing: [epsilon,decay_rate]
#Greedy: None
AlgorithmDependendVariable=[100, 0.999 , 'yes' ,np.ones(10)*1000]

#When you do ExploreThenCommit, you can choose the number of exploration steps

######### END INPUT BY THE USER ################################################################################################






#Initialize the Bandit
bandit=Bandit(num_arms, distribution, means, random_mode, reward_gap)
#Determine the optimale length of the exploration phase
means = bandit.__repr__()  # Means of the arms
max_mean = np.max(means)  # Highest mean
if AlgorithmDependendVariable == None and ChosenAgent == 'ExploreThenCommit':
    explore_steps=bandit.optimize(num_timesteps)
    AlgorithmDependendVariable=explore_steps
else:
    explore_steps=None

# Initialize the Agent


# Dictionary mapping agent names to their respective classes
AgentClasses = {
    'ExploreThenCommit': ExploreThenCommit,
    'EpsilonGreedy': EpsilonGreedy,
    'UCB': UCB,
    'Greedy': Greedy,
    'EpsilonDecreasing': EpsilonDecreasing,
    'PolicyGradient': PolicyGradient,
    "Boltzmann": Boltzmann
}

# Check if the chosen agent is valid

if ChosenAgent in AgentClasses:
    agent = AgentClasses[ChosenAgent](num_arms,AlgorithmDependendVariable)
else:
    raise ValueError(f"Invalid agent chosen. Choose from {list(AgentClasses.keys())}")

print(f"Chosen Agent: {ChosenAgent}, Exploration Steps: {explore_steps}")


#Initialize the Game
game = Game(num_arms, distribution, means, random_mode, reward_gap, explore_steps, num_timesteps, Iterations,AlgorithmDependendVariable)


#Play the Game once
#df=game.GameOnce( num_arms, means, num_timesteps, bandit, max_mean, agent)

#Play the Game n times and average the results
df=game.AverageGame(bandit, max_mean, means, agent)





# --- Plotting ---
estimated_means_var_array = np.array(df['estimated_means_var'].tolist())
probabilities_var_array = np.array(df['probabilities_var'].tolist())

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(df['time'], df['regret_var'], label="Regret Variance", color='blue')
plt.xlabel("Time")
plt.ylabel("Variance")
plt.title("Variance of Regret over Time")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(df['time'], df['correct_action_rate_var'], label="Correct Action Rate Variance", color='green')
plt.xlabel("Time")
plt.ylabel("Variance")
plt.title("Variance of Correct Action Rate over Time")
plt.legend()

plt.subplot(2, 2, 3)
for arm in range(num_arms):
    plt.plot(df['time'], estimated_means_var_array[:, arm], label=f'Arm {arm} Variance', alpha=0.6)
plt.xlabel("Time")
plt.ylabel("Variance")
plt.title("Variance of Estimated Means per Arm")
plt.legend()

plt.subplot(2, 2, 4)
for arm in range(num_arms):
    plt.plot(df['time'], probabilities_var_array[:, arm], label=f'Arm {arm} Variance', alpha=0.6)
plt.xlabel("Time")
plt.ylabel("Variance")
plt.title("Variance of Selection Probability per Arm")
plt.legend()

plt.tight_layout()
plt.show()



# --- Plotting ---
# Convert lists of arrays to numpy arrays for easier plotting
estimated_means_array = np.array(df['estimated_means'].tolist())  # Shape (num_timesteps, num_arms)
probabilities_array = np.array(df['probabilities'].tolist())  # Shape (num_timesteps, num_arms)

# --- (a) Regret over time ---
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(df['time'], df['regret'], label=" Pointwise Expected Regret", color='blue')
plt.plot(df['time'], df['regret_CI_upper'], label="Upper Confidence Bound", color='Red')
plt.xlabel("Time")
plt.ylabel("Regret")
plt.title("Regret over Time")
plt.legend()

# --- (b) Correct action rate over time ---
plt.subplot(2, 2, 2)
plt.plot(df['time'], df['correct_action_rate'], label="Correct Action Rate", color='green')
plt.xlabel("Time")
plt.ylabel("Correct Action Rate")
plt.title("Correct Action Rate over Time")
plt.legend()

# --- (c) Estimated means vs actual means ---
plt.subplot(2, 2, 3)
for arm in range(num_arms):
    plt.plot(df['time'], estimated_means_array[:, arm], label=f'Estimated Arm {arm}', alpha=0.6)
for arm in range(num_arms):
    plt.axhline(y=means[arm], color='red', linestyle='dotted', label=f'Actual Arm {arm}')
plt.xlabel("Time")
plt.ylabel("Estimated Means")
plt.title(f"Estimated Means vs Actual Means (Best arm is {np.argmax(means)}) ")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)

# --- (d) Average probabilities of choosing each arm ---
plt.subplot(2, 2, 4)
for arm in range(num_arms):
    plt.plot(df['time'], probabilities_array[:, arm], label=f'Arm {arm}', alpha=0.6)
plt.xlabel("Time")
plt.ylabel("Probability")
plt.title("Average Selection Probability per Arm")
plt.legend()

plt.tight_layout()
plt.show()