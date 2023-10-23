import numpy as np
import matplotlib.pyplot as plt

def load(filename):
    return np.load(filename)

def plot(filenames, labels):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    for i in range(len(filenames)):
        data = load(filenames[i])
        data = smooth(data, 100)
        axs[i].plot(data, label=labels[i], color='C'+str(i))
        axs[i].set_xlabel('episode')
        axs[i].legend()
    plt.show()

def smooth(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def main():
    filenames = ['rewards_test_experiment.npy', 'valueloss_test_experiment.npy']
    labels = ['rewards', 'value loss']
    plot(filenames, labels)

if __name__ == "__main__":
    main()