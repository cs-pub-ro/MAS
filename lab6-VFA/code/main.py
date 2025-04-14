import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import json
import tqdm
import random

class CosineActivation(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cos(x)

class Estimator(object):
    def __init__(self, state_dim = 4, action_dim = 2, hidden_dim = 100, lr = 0.0001, activation = 'cos'):

        if activation == 'cos':
            activation = CosineActivation()
        elif activation == 'sigmoid':
            activation = torch.nn.Sigmoid()
        elif activation == 'tanh':
            activation = torch.nn.Tanh()

        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
                        torch.nn.Linear(state_dim, hidden_dim),
                        activation,
                        torch.nn.Linear(hidden_dim, action_dim)
                )
        
        # the first weight is a state_dim x hidden_dim matrix. 
        # Initialize each row with a normal distribution with mean 0 and standard deviation sqrt((i+1) * 0.5), where i is the row index.
        # the bias is uniformly distributed between 0 and 2 pi
        for i in range(state_dim):
            torch.nn.init.normal_(self.model[0].weight[i], mean = 0, std = np.sqrt((i+1) * 0.5))
        torch.nn.init.uniform_(self.model[0].bias, a = 0, b = 2 * np.pi)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)

    def update(self, state, y):
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, torch.Tensor(y))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, state):
            with torch.no_grad():
                return self.model(torch.Tensor(state))


def q_learning(
    env,
    model,
    episodes,
    decay = False,
    gamma = 0.9,
    epsilon = 0.1,
    eps_decay = 0.99,
):

    total_reward = []
    total_loss = []

    for episode in tqdm.tqdm(range(episodes)):
        state, _ = env.reset()

        done = False
        episode_reward = 0

        while not done:

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model.predict(state)
                action = torch.argmax(q_values).item()

            step_res = env.step(action)
            next_state, reward, done, _, _ = step_res

            episode_reward += reward

            q_values = model.predict(state).tolist()

            if done:
                q_values[action] = reward
                loss = model.update(state, q_values)
                total_loss.append(loss)
                break

            q_values_next = model.predict(next_state)
            q_values[action] = reward + gamma * torch.max(q_values_next).item()
            loss = model.update(state, q_values)
            total_loss.append(loss)

            state = next_state

        # Update epsilon
        if decay:
            epsilon = max(epsilon * eps_decay, 0.001)

        total_reward.append(episode_reward)

    return total_reward, total_loss


def moving_average_with_variance(data, window_size=50):
    """Calculate moving average and variance over the given window size."""
    if len(data) < window_size:
        return [], [], []
    
    indices = np.arange(window_size - 1, len(data))
    means = []
    upper_bounds = []
    lower_bounds = []
    
    for i in range(window_size - 1, len(data)):
        window = data[i - window_size + 1 : i + 1]
        mean_val = np.mean(window)
        std_val = np.std(window)
        means.append(mean_val)
        upper_bounds.append(mean_val + std_val)
        lower_bounds.append(mean_val - std_val)
    
    return indices, means, [lower_bounds, upper_bounds]


spec = {
    'activation': 'sigmoid',
    'epsilon': 0.05,
    'gamma': 1.0,
    'lr': 0.0005,
    'episodes': 1000,
    'decay': True
}

if __name__ == '__main__':
    # make a CartPole environment with a seed of 42 and maximum episode length of 200
    env = gym.make("CartPole-v1", max_episode_steps=100)
    env.reset(seed=42)

    estimator = Estimator(
        state_dim = env.observation_space.shape[0],
        action_dim = env.action_space.n,
        hidden_dim = 100,
        lr = spec['lr']
    )
    reward, total_loss = q_learning(
        env,
        estimator,
        spec['episodes'],
        gamma = spec['gamma'],
        epsilon = spec['epsilon'],
        decay = spec['decay']
    )

    # dump the spec dict into a key value string
    spec_str = '_'.join([f'{k}={v}' for k, v in spec.items()])

    with open(f'dumps/experiment_{spec_str}.json', 'wt') as f:
        json.dump({
            'reward': reward,
            'total_loss': total_loss,
            'spec': spec
            }, f)
        
    # close the environment
    env.close()

    # plot the results, showing loss and reward on a plot with two subplots and displaying a moving average over 50 episodes
    # Create the figure and subplots with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    print(f"Reward length: {len(reward)}")
    print(f"Loss length: {len(total_loss)}")

    # Process and plot reward data
    x_reward, reward_mean, reward_var = moving_average_with_variance(np.array(reward), window_size=50)
    ax1.plot(x_reward, reward_mean, color='blue', label='Moving Avg (50 episodes)')
    ax1.fill_between(x_reward, reward_var[0], reward_var[1], alpha=0.3, color='blue')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Moving Average of Total Reward with Variance')
    ax1.yaxis.set_major_locator(plt.MultipleLocator(100))
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Process and plot loss data
    x_loss, loss_mean, loss_var = moving_average_with_variance(np.array(total_loss), window_size=500)
    ax2.plot(x_loss, loss_mean, color='red', label='Moving Avg (500 episodes)')
    ax2.fill_between(x_loss, loss_var[0], loss_var[1], alpha=0.3, color='red')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Total Loss')
    ax2.set_title('Moving Average of Total Loss with Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()