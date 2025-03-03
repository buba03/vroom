import matplotlib.pyplot as plt
from IPython import display
import os

plt.ion()


def plot(scores, mean_scores, save_filename='training_plot.png'):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean Score")

    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

    plt.legend()
    plt.grid(True)

    # Save as an image
    plt.savefig(os.path.join('plots', save_filename))

    plt.show(block=False)
    plt.pause(0.1)
