import numpy as np
import matplotlib.pyplot as plt
import os


def plot(names, smooth_data=False, window_size=100):
    
    # Create plot per trained model and average over all runs and also show the standard deviation
    # number of runs can be different for each model
    fig, axs = plt.subplots(3, 1, figsize=(15, 6))

    for name in names:

        fig2, axs2 = plt.subplots(3, 1, figsize=(15, 6))
        runs = os.listdir('results/'+name+'/train_logs/')

        rewards = np.array([np.load('results/'+name+'/train_logs/'+str(run)+'/rewards.npy') for run in runs])
        value_losses = np.array([np.load('results/'+name+'/train_logs/'+str(run)+'/valueloss.npy') for run in runs])
        policy_losses = np.array([np.load('results/'+name+'/train_logs/'+str(run)+'/policyloss.npy') for run in runs])

        if smooth_data:
            rewards = np.array([smooth(reward, window_size) for reward in rewards])
            value_losses = np.array([smooth(value_loss, window_size) for value_loss in value_losses])
            policy_losses = np.array([smooth(policy_loss, window_size) for policy_loss in policy_losses])

        mean_rewards, std_rewards = np.mean(rewards, axis=0), np.std(rewards, axis=0)
        mean_value_losses, std_value_losses = np.mean(value_losses, axis=0), np.std(value_losses, axis=0)
        mean_policy_losses, std_policy_losses = np.mean(policy_losses, axis=0), np.std(policy_losses, axis=0)

        axs2[0].plot(mean_rewards, label='mean rewards', color='C0')
        axs2[1].plot(mean_value_losses, label='mean value loss', color='C1')
        axs2[2].plot(mean_policy_losses, label='mean policy loss', color='C2')
        axs2[0].fill_between(range(len(mean_rewards)), mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2, color='C0')
        axs2[1].fill_between(range(len(mean_value_losses)), mean_value_losses-std_value_losses, mean_value_losses+std_value_losses, alpha=0.2, color='C1')
        axs2[2].fill_between(range(len(mean_policy_losses)), mean_policy_losses-std_policy_losses, mean_policy_losses+std_policy_losses, alpha=0.2, color='C2')

        axs2[0].legend()
        axs2[1].legend()
        axs2[2].legend()

        # set common x label
        fig2.text(0.5, 0.04, 'time step', ha='center')

        axs2[0].set_ylabel('reward')
        axs2[1].set_ylabel('value loss')
        axs2[2].set_ylabel('policy loss')

        axs2[0].set_ylim(-8, 0)
        axs2[1].set_ylim(0, 1000)
        axs2[2].set_ylim(-150,150)
        plt.tight_layout()
        os.makedirs(f'results/{name}/plots', exist_ok=True)
        plt.savefig(f'results/{name}/plots/plot.png')
        plt.close(fig2)

        axs[0].plot(mean_rewards, label=name)
        axs[1].plot(mean_value_losses, label=name)
        axs[2].plot(mean_policy_losses, label=name)
        axs[0].fill_between(range(len(mean_rewards)), mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2)
        axs[1].fill_between(range(len(mean_value_losses)), mean_value_losses-std_value_losses, mean_value_losses+std_value_losses, alpha=0.2)
        axs[2].fill_between(range(len(mean_policy_losses)), mean_policy_losses-std_policy_losses, mean_policy_losses+std_policy_losses, alpha=0.2)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    # set common x label
    fig.text(0.5, 0.04, 'time step', ha='center')

    axs[0].set_ylabel('reward')
    axs[1].set_ylabel('value loss')
    axs[2].set_ylabel('policy loss')

    axs[0].set_ylim(-8, 0)
    axs[1].set_ylim(0, 1000)
    axs[2].set_ylim(-150,150)
    plt.tight_layout()
    plt.savefig('results/all_plots.png')


def smooth(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def main():

    names = os.listdir('results/')
    names.remove('all_plots.png')

    smooth_data = True
    window_size = 100
    plot(names, smooth_data, window_size)


if __name__ == "__main__":
    main()