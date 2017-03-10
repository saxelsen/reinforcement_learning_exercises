import numpy as np

np.random.seed(123)
steps = 10
k_bandits = 10
explore_factor = 0.5
alpha = 0.05
arange = np.arange(k_bandits)

# Initialize arrays
real_mean_q = np.zeros((steps, k_bandits))
N_taken_avg = np.zeros((steps, k_bandits))
mean_Q_avg = np.zeros((steps, k_bandits))
mean_Q_exp_avg = np.zeros((steps, k_bandits))
rewards = np.zeros((steps, k_bandits))
score_avg = np.zeros((steps, 1))
score_exp_avg = np.zeros((steps, 1))
score_optimal = np.zeros((steps, 1))
action_avg = np.zeros((steps, 1))
action_exp_avg = np.zeros((steps, 1))

# Run first period
rewards[0] = real_mean_q[0] + np.random.randn(k_bandits)
action_avg[0] = np.random.choice(arange)
action_exp_avg[0] = action_avg[0]
action_avg_mask = (arange == action_avg[0])
N_taken_avg[0] = N_taken_avg[0] + action_avg_mask * np.ones((1, k_bandits))
score_avg[0] = rewards[0, int(action_avg[0])]
score_exp_avg[0] = rewards[0, int(action_exp_avg[0])]
score_optimal[0] = rewards[0, int(action_avg[0])]
mean_Q_avg[0] = action_avg_mask * rewards[0]
mean_Q_exp_avg[0] = action_avg_mask * rewards[0]

for step in range(1, steps):
    real_mean_q[step] = real_mean_q[step - 1] + np.random.randn(k_bandits)
    rewards[step] = real_mean_q[step] + np.random.randn(k_bandits)

    greedy = np.random.randn(1) <= 1 - explore_factor
    if greedy:
        action_avg[step] = np.where(mean_Q_avg[step - 1] == np.max(mean_Q_avg[step - 1]))[0][0]
        action_exp_avg[step] = np.where(mean_Q_exp_avg[step - 1] == np.max(mean_Q_exp_avg[step - 1]))[0][0]
    else:
        action_avg[step] = np.random.choice(np.where(mean_Q_avg[step - 1] != np.max(mean_Q_avg[step - 1]))[0])
        action_exp_avg[step] = np.random.choice(np.where(mean_Q_exp_avg[step - 1] != np.max(mean_Q_exp_avg[step - 1]))[0])

    score_avg[step] = rewards[step, int(action_avg[step])]
    score_exp_avg[step] = rewards[step, int(action_exp_avg[step])]
    score_optimal[step] = rewards[step, int(np.where(real_mean_q[step] == np.max(real_mean_q[step]))[0])]

    action_avg_mask = (arange == action_avg[step])
    action_exp_avg_mask = (arange == action_exp_avg[step])

    N_taken_avg[step] = N_taken_avg[step - 1] + action_avg_mask * np.ones((1, k_bandits))
    mean_Q_avg[step] = mean_Q_avg[step - 1] + action_avg_mask * np.minimum(1, 1/N_taken_avg[step]) * (rewards[step] - mean_Q_avg[step - 1])
    mean_Q_exp_avg[step] = mean_Q_exp_avg[step - 1] + alpha * action_exp_avg_mask * (rewards[step] - mean_Q_exp_avg[step - 1])

print("Final mean rewards: ")
print(real_mean_q[-1])
print("Final Q_avg: ")
print(mean_Q_avg[-1])
print("Final Q_exp_avg: ")
print(mean_Q_exp_avg[-1])
print("Final score avg:")
print(score_avg[-1])
print("Final score exp_avg:")
print(score_exp_avg[-1])
print("Final optimal score:")
print(score_optimal[-1])