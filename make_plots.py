import numpy as np
import matplotlib.pyplot as plt
import os


def plot(filenames, labels):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    for filename in filenames:
        for i in range(len(labels)):
            data = np.load(labels[i]+filename+'.npy')
            data = smooth(data, 100)
            axs[i].plot(data, label=labels[i]+filename, lw=0.75)
    axs[0].legend()
    axs[1].legend()
    axs[0].set_xlabel('time step')
    axs[1].set_xlabel('time step')
    axs[0].set_ylabel('reward')
    axs[1].set_ylabel('value loss')
    axs[1].set_ylim(0, 20000)

    if not os.path.exists('results/plots'):
        os.makedirs('results/plots')
    file_names = sorted(os.listdir('results/plots'))
    if len(file_names) > 0:
        new_name = 'experiment_' + str(int(file_names[-1][11:-4])+1)
    else:
        new_name = 'experiment_1'

    plt.tight_layout()
    plt.savefig('results/plots/' + str(new_name) + '.png')

def smooth(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def main():
    filenames = os.listdir()
    filenames = [filename[7:-4] for filename in filenames if filename[-4:] == '.npy' and filename[:7] == 'rewards']
    types = ['rewards', 'valueloss']
    plot(filenames, types)

if __name__ == "__main__":
    main()
