from Classes import Bandit, Game, ExploreThenCommit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


#Paraemters
num_arms=10
distribution='gaussian'
means=None
random_mode=False
reward_gap=None
num_timesteps = 10000  
Iterations=10
ArbitraryExplorationSteps=None #Choose either None for automatic optimization or a number for a fixed number of exploration steps


#Initialize the Bandit
bandit=Bandit(num_arms, distribution, means, random_mode, reward_gap)


#Determine the optimale length of the exploration phase
means = bandit.__repr__()  # Means of the arms
max_mean = np.max(means)  # Highest mean
explore_steps=bandit.optimize(ArbitraryExplorationSteps,max_mean,num_timesteps)


#Initialize the Agent
agent = ExploreThenCommit(num_arms, explore_steps)


#Initialize the Game
game = Game(num_arms, distribution, means, random_mode, reward_gap, explore_steps, num_timesteps, Iterations)


#Play the Game once
#df=game.GameOnce( num_arms, means, num_timesteps, bandit, max_mean, agent)

#Play the Game n times and average the results
df=game.AverageGame(bandit, max_mean, means, agent)



# --- Plotting ---
# Convert lists of arrays to numpy arrays for easier plotting
estimated_means_array = np.array(df['estimated_means'].tolist())  # Shape (num_timesteps, num_arms)
probabilities_array = np.array(df['probabilities'].tolist())  # Shape (num_timesteps, num_arms)

# --- (a) Regret over time ---
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(df['time'], df['regret'], label="Regret", color='blue')
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
