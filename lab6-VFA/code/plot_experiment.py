import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path


def moving_average_with_variance(data, window_size=50):
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


def plot_experiment(dump_path: Path):
    with open(dump_path) as f:
        data = json.load(f)

    reward = data['reward']
    total_loss = data['total_loss']
    spec = data.get('spec', {})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    spec_str = ', '.join(f'{k}={v}' for k, v in spec.items())
    fig.suptitle(spec_str, fontsize=9)

    x_reward, reward_mean, reward_var = moving_average_with_variance(np.array(reward), window_size=50)
    ax1.plot(x_reward, reward_mean, color='blue', label='Moving Avg (50 episodes)')
    ax1.fill_between(x_reward, reward_var[0], reward_var[1], alpha=0.3, color='blue')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Moving Average of Total Reward with Variance')
    reward_tick = (max(reward) - min(reward)) / 20
    ax1.yaxis.set_major_locator(plt.MultipleLocator(reward_tick))
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    x_loss, loss_mean, loss_var = moving_average_with_variance(np.array(total_loss), window_size=500)
    ax2.plot(x_loss, loss_mean, color='red', label='Moving Avg (500 steps)')
    ax2.fill_between(x_loss, loss_var[0], loss_var[1], alpha=0.3, color='red')
    loss_tick = (max(total_loss) - min(total_loss)) / 20
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Total Loss')
    ax2.set_title('Moving Average of Total Loss with Variance')
    ax2.yaxis.set_major_locator(plt.MultipleLocator(loss_tick))
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = dump_path.with_suffix('.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render graphs from an experiment dump and save as PNG.')
    parser.add_argument('dump', metavar='DUMP', type=Path, help='Path to the experiment JSON dump file')
    args = parser.parse_args()

    plot_experiment(args.dump)
