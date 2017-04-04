import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

np.set_printoptions(precision=3, linewidth=200)

np.random.seed(123)
reps = 1000
steps = 1000
k = 10
epsilon = 0.1
alpha = 0.3

true_mean = np.zeros((reps, k))
average_estimates = np.zeros((reps, k))
alpha_estimates = np.zeros((reps, k))
N = np.zeros((reps, k))
starting_action = np.random.randint(0, k-1, size=(reps,))

best_rewards = np.zeros(steps)
average_rewards = np.zeros(steps)
alpha_rewards = np.zeros(steps)

for i in range(0, steps):
    rewards = true_mean + np.random.randn(reps, k)
    print('****** i={} *******'.format(i))
    print('Average true means: {}'.format(true_mean.mean(axis=0)))
    print('Average rewards: {}'.format(rewards.mean(axis=0)))
    print('Sample average estimator: {}'.format(average_estimates.mean(axis=0)))
    print('Moving average estimator: {}'.format(alpha_estimates.mean(axis=0)))
    print('')

    is_greedy = np.random.rand(reps) < 1-epsilon

    if i == 0:
        average_action = starting_action
        alpha_action = starting_action
    else:
        average_action = is_greedy * np.argmax(average_estimates, axis=1) + ~is_greedy * np.random.randint(0, k - 1, size=(reps,))
        alpha_action = is_greedy * np.argmax(alpha_estimates, axis=1) + ~is_greedy * np.random.randint(0, k - 1, size=(reps,))

    average_action_mask = np.zeros((reps, k))
    alpha_action_mask = np.zeros((reps, k))
    average_action_mask[np.arange(0, reps), average_action] = 1
    alpha_action_mask[np.arange(0, reps), alpha_action] = 1

    N += average_action_mask
    average_estimates += average_action_mask * np.divide(1, N, out=np.zeros_like(N), where=N != 0) * (rewards - average_estimates)
    alpha_estimates += alpha_action_mask * alpha * (rewards - alpha_estimates)

    true_mean -= np.random.randn(reps, k)  # Update the moving means

    best_rewards[i] = rewards.max(axis=1).mean()
    average_rewards[i] = (average_action_mask * rewards).sum(axis=1).mean()
    alpha_rewards[i] = (alpha_action_mask * rewards).sum(axis=1).mean()

x = np.arange(1, steps+1)

fig, ax = plt.subplots()

ax.plot(x, best_rewards, 'g', label='best reward')
ax.plot(x, average_rewards, 'b', label='sample average')
ax.plot(x, alpha_rewards, 'r', label='moving average')
ax.legend(loc='upper right', shadow=True)
plt.show()
