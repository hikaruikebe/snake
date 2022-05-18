import matplotlib.pyplot as plt

# from IPython import display

plt.ion()


def plot(scores, mean_scores, moving_mean_scores):
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.plot(moving_mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.text(
        len(moving_mean_scores) - 1, moving_mean_scores[-1], str(moving_mean_scores[-1])
    )
    plt.legend(["Score", "Total Average", "Moving Average"])
    plt.show(block=False)
    plt.pause(0.1)
