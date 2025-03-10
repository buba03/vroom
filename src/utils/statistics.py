import matplotlib.pyplot as plt
import os


def training_plot(scores, mean_scores, save_filename='training_plot.png'):
    plt.clf()

    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean Score")

    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

    plt.legend()
    plt.grid(True)

    # Save as an image
    plt.savefig(os.path.join('plots', save_filename))

    plt.show(block=False)
    plt.pause(0.1)


def debug_plot(epsilon, steps_taken, save_filename='debug_plot.png'):
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
