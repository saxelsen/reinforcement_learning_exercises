import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

np.random.seed(123)
reps = 1000
steps = 1000
k_bandits = 10
explore_factors = [0, 0.01, 0.1]

true_mean = np.random.randn(k_bandits)
all_avg_rewards = np.zeros((3, steps))


def take_action(step, rep, explore, step_rewards, action_values, avg_rewards, n_actions):
    is_greedy = np.random.rand(1) <= 1 - explore

    if step == 1:
        action = starting_action
    elif is_greedy:
        action = np.random.choice(np.where(action_values == np.max(action_values))[0])
    else:
        action = np.random.choice(np.where(action_values != np.max(action_values))[0])

    action_reward = step_rewards[action]
    avg_rewards[step - 1] += 1 / rep * (action_reward - avg_rewards[step - 1])
    n_actions[action] += 1
    n = n_actions[action]
    action_values[action] += 1 / n * (action_reward - action_values[action])

for a_rep in range(1, reps + 1):
    starting_action = np.random.choice(np.arange(0, k_bandits))
    all_action_values = np.zeros((3, k_bandits))
    all_n_actions = np.zeros((3, k_bandits))

    for step in range(1, steps + 1):
        rewards = true_mean + np.random.randn(k_bandits)

        for i in range(0, 3):
            take_action(step, a_rep, explore_factors[i], rewards, all_action_values[i], all_avg_rewards[i], all_n_actions[i])

print(true_mean)
x = np.arange(1, steps+1)

fig, ax = plt.subplots()

ax.plot(x, all_avg_rewards[0], 'b', label='epsilon 0.00')
ax.plot(x, all_avg_rewards[1], 'r', label='epsilon 0.01')
ax.plot(x, all_avg_rewards[2], 'y', label='epsilon 0.10')
ax.legend(loc='upper right', shadow=True)
plt.show()