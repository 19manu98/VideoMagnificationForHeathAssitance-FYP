import matplotlib.pyplot as plt
import time

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
    last = len(dataframe['x'])-1 # get the last values of the dataframe containing the valuee
    # plt.setp(fig, color='r')
    axes.set_xlim(dataframe['x'][last]-100,dataframe['x'][last]) #display just 100 values
    axes.lines[0].set_data(dataframe['x'],dataframe['green']) #set the plotting data
    axes.relim() #recompute the datalimits
    axes.autoscale_view() #autoscale the view
    fig.canvas.flush_events() #flush the event (update)
    time.sleep(0.00001) #wait for the next interaction

def plot(dataframe,fig,axes):
    last = len(dataframe['x'])-1 # get the last values of the dataframe containing the values
    title = str(round(dataframe['bpm'][last],0)) + ' BPM'
    fig_t=fig.suptitle(title)
    plt.setp(fig_t, color='r')
    axes.set_xlim(dataframe['x'][last]-100,dataframe['x'][last]) #display just 100 values
    axes.lines[0].set_data(dataframe['x'],dataframe['bpm']) #set the plotting data
    axes.relim() #recompute the datalimits
    axes.autoscale_view() #autoscale the view
    fig.canvas.flush_events() #flush the event (update)
    time.sleep(0.00001) #wait for the next interaction