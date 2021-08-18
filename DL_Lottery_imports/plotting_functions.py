
import matplotlib.pyplot as plt
from numpy import (
    ndarray,
    mean,
    vstack
)


def plot_reward_components(
        reward_components_per_episode: list
) -> None:
    fig, ax = plt.subplots()
    for key in reward_components_per_episode[0].keys():
        values = [x[key] for x in reward_components_per_episode]
        mean_values = mean(vstack(values), axis=1)
        ax.scatter(range(len(mean_values)), mean_values)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean Reward')
    ax.legend(reward_components_per_episode[0].keys())

    ax.grid(alpha=.25)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.tight_layout()


def plot_scatter_plot(
        data: ndarray,
        title: str
) -> None:
    fig, ax = plt.subplots(1)
    ax.scatter(range(len(data)), data)

    ax.set_ylabel(title)
    ax.set_xlabel('Episode')

    ax.grid(alpha=.25)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.tight_layout()


def plot_bar_plot(
        data: ndarray,
        title: str
) -> None:
    fig, ax = plt.subplots(1)
    ax.bar(range(len(data)), data)

    ax.set_ylabel(title)
    ax.set_xlabel('Episode')

    ax.grid(alpha=.25)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.tight_layout()


def plot_time_histogram(
        data: list or ndarray,
        bins: int,
        title: str
) -> None:
    bin_size = len(data) // bins
    bin_values = []
    for bin_id in range(bins):
        bin_values.append(sum(data[bin_id * bin_size:(bin_id + 1) * bin_size]) / bin_size)

    fig, ax = plt.subplots()
    ax.bar(range(bins), bin_values)

    ax.set_ylabel(title)
    ax.set_xlabel('Training')

    ax.grid(alpha=.25)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.tight_layout()
