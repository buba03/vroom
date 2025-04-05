""" Module for plotting useful data during training. """

import os
import matplotlib.pyplot as plt


def training_plot(scores, mean_scores, rewards, mean_rewards, save_filename='training_plot.png'):
    """
    Plots the training progress, showing scores and mean scores over time.
    Saves the plot after each call, overwriting the previous if necessary.

    :param scores: List of scores.
    :param mean_scores: List of mean scores.
    :param save_filename: Filename (inside the 'plots' folder) to save the plot as an image. Default is 'training_plot.png'
    """
    plt.clf()

    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean Score")
    plt.plot(rewards, label="Rewards")
    plt.plot(mean_rewards, label="Mean Rewards")

    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.text(len(rewards) - 1, rewards[-1], str(rewards[-1]))
    plt.text(len(mean_rewards) - 1, mean_rewards[-1], str(mean_rewards[-1]))
    plt.text(len(rewards) - 1, rewards[-1], str(rewards[-1]))
    plt.text(len(mean_rewards) - 1, mean_rewards[-1], str(mean_rewards[-1]))

    plt.legend()
    plt.grid(True)

    # Save as an image
    plt.savefig(os.path.join('plots', save_filename))

    plt.show(block=False)
    plt.pause(0.1)


def debug_plot(epsilon, steps_taken, save_filename='debug_plot.png'):
    """
    Plots debug values such as epsilon and steps taken per game.
    Saves the plot after each call, overwriting the previous if necessary.

    :param epsilon: List of epsilon values.
    :param steps_taken: List of steps taken in each game.
    :param save_filename: Filename (inside the 'plots' folder) to save the plot as an image. Default is 'debug_plot.png'
    """
    plt.clf()

    plt.title('Debug Values')
    plt.xlabel('Number of Games')
    plt.ylabel('Values')

    plt.plot(epsilon, label="Epsilon")
    plt.plot(steps_taken, label="Steps")

    plt.ylim(ymin=0, ymax=1)
    plt.xlim(xmin=0)
    plt.text(len(epsilon) - 1, epsilon[-1], str(epsilon[-1]))
    plt.text(len(steps_taken) - 1, steps_taken[-1], str(steps_taken[-1]))

    plt.legend()
    plt.grid(True)

    # Save as an image
    plt.savefig(os.path.join('plots', save_filename))
