import numpy as np
import matplotlib.pyplot as plt
import os

def load(filename):
    return np.load(filename)

def plot(filenames, labels):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    for i in range(len(filenames)):
        data = load(filenames[i])
        data = smooth(data, 10000)
        axs[i].plot(data, label=labels[i], color='C'+str(i))
        axs[i].set_xlabel('time step')
        axs[i].legend()

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
    filenames = ['rewards_experiment_10000000.npy', 'valueloss_experiment_10000000.npy']
    labels = ['rewards', 'value loss']
    plot(filenames, labels)

if __name__ == "__main__":
    main()