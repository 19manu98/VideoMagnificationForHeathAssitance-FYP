import matplotlib.pyplot as plt

def create_bpm_plot():
    plt.ion()
    plt.rcParams['axes.facecolor'] = 'black'
    fig = plt.figure(figsize=(16, 8),num=0)
    fig.patch.set_facecolor('black')
    axes = plt.subplot(1, 1, 1)
    axes.set_xlabel('Time')
    axes.set_ylabel('BPM')
    t = []
    y = []
    axes.plot(t, y,color='g')
    return axes, fig


def create_green_plot():
    plt.ion()
    fig = plt.figure(figsize=(16, 8), num=1)
    axes = plt.subplot(1, 1, 1)
    axes.set_xlabel('Frame')
    axes.set_ylabel('Green')
    t = []
    y = []
    axes.plot(t, y, color='g')
    return axes, fig

def plot_green(dataframe,fig,axes):
    plt.plot(dataframe['x'],dataframe['green'])
    plt.show()


def plot(dataframe,fig,axes):
    plt.plot(dataframe['x'],dataframe['bpm'])
    plt.show()