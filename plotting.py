import matplotlib.pyplot as plt

def plot(dataframe):
    plt.clf()
    # Add titles
    plt.title("Change of the different channels")
    plt.xlabel("Frames")
    plt.ylabel("Channel value")
    plt.ion()
    plt.show()
    # style
    plt.style.use('seaborn-darkgrid')

    # palette
    palette = plt.get_cmap('Set1')

    # multiple line plot
    num = 0
    for column in dataframe.drop('x', axis=1):
        num += 1
        plt.plot(dataframe['x'], dataframe[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

    # Add legend
    plt.legend(loc=2, ncol=2)

    plt.draw()
    plt.pause(1)
    # print(len(peaks[0]))
    # print(times[-1]-times[0])
