import numpy as np
import matplotlib.pyplot as plt
import os

def plot(names, smooth_data=False, window_size=100):
    
    value_loss_coef = 0.5
    entropy_coef = 0.05
    seac_coef = 1.0

    # Create plot per trained model and average over all runs and also show the standard deviation
    # number of runs can be different for each model
    fig, axs = plt.subplots(4, 1, figsize=(30, 12))

    for name in names:

        fig2, axs2 = plt.subplots(4, 1, figsize=(15, 6))
        runs = os.listdir('results/'+name+'/train_logs/')

        rewards = np.array([np.load('results/'+name+'/train_logs/'+str(run)+'/rewards.npy') for run in runs])
        policy_losses = np.array([np.load('results/'+name+'/train_logs/'+str(run)+'/policyloss.npy') for run in runs])
        value_losses = np.array([np.load('results/'+name+'/train_logs/'+str(run)+'/valueloss.npy') for run in runs])
        dist_entropies = np.array([np.load('results/'+name+'/train_logs/'+str(run)+'/distentropy.npy') for run in runs])
        # importance_samplings = np.array([np.load('results/'+name+'/train_logs/'+str(run)+'/importancesampling.npy') for run in runs])
        seac_policy_losses = np.array([np.load('results/'+name+'/train_logs/'+str(run)+'/seacpolicyloss.npy') for run in runs])
        seac_value_losses = np.array([np.load('results/'+name+'/train_logs/'+str(run)+'/seacvalueloss.npy') for run in runs])

        total_loss = policy_losses \
            + value_loss_coef * value_losses \
            - entropy_coef * dist_entropies \
            + seac_coef * seac_policy_losses \
            + seac_coef * value_loss_coef * seac_value_losses

        if smooth_data:
            rewards = np.array([smooth(reward, window_size) for reward in rewards])
            value_losses = np.array([smooth(value_loss, window_size) for value_loss in value_losses])
            policy_losses = np.array([smooth(policy_loss, window_size) for policy_loss in policy_losses])
            total_loss = np.array([smooth(loss, window_size) for loss in total_loss])

        mean_rewards, std_rewards = np.mean(rewards, axis=0), np.std(rewards, axis=0)
        mean_value_losses, std_value_losses = np.mean(value_losses, axis=0), np.std(value_losses, axis=0)
        mean_policy_losses, std_policy_losses = np.mean(policy_losses, axis=0), np.std(policy_losses, axis=0)
        mean_total_loss, std_total_loss = np.mean(total_loss, axis=0), np.std(total_loss, axis=0)

        axs2[0].plot(mean_rewards, label='mean rewards', color='C0')
        axs2[1].plot(mean_value_losses, label='mean value loss', color='C1')
        axs2[2].plot(mean_policy_losses, label='mean policy loss', color='C2')
        axs2[3].plot(mean_total_loss, label='mean total loss', color='C3')
        axs2[0].fill_between(range(len(mean_rewards)), mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2, color='C0')
        axs2[1].fill_between(range(len(mean_value_losses)), mean_value_losses-std_value_losses, mean_value_losses+std_value_losses, alpha=0.2, color='C1')
        axs2[2].fill_between(range(len(mean_policy_losses)), mean_policy_losses-std_policy_losses, mean_policy_losses+std_policy_losses, alpha=0.2, color='C2')
        axs2[3].fill_between(range(len(mean_total_loss)), mean_total_loss-std_total_loss, mean_total_loss+std_total_loss, alpha=0.2, color='C3')

        axs2[0].legend()
        axs2[1].legend()
        axs2[2].legend()
        axs2[3].legend()

        # set common x label
        fig2.text(0.5, 0.04, 'time step', ha='center')

        axs2[0].set_ylabel('reward')
        axs2[1].set_ylabel('value loss')
        axs2[2].set_ylabel('policy loss')
        axs2[3].set_ylabel('total loss')

        # axs2[0].set_ylim(-8, 0)
        # axs2[1].set_ylim(0, 1000)
        # axs2[2].set_ylim(-150,150)
        plt.tight_layout()
        os.makedirs(f'results/{name}/plots', exist_ok=True)
        plt.savefig(f'results/{name}/plots/plot.png')
        plt.close(fig2)

        axs[0].plot(mean_rewards, label=name)
        axs[1].plot(mean_value_losses, label=name)
        axs[2].plot(mean_policy_losses, label=name)
        axs[3].plot(mean_total_loss, label=name)
        axs[0].fill_between(range(len(mean_rewards)), mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2)
        axs[1].fill_between(range(len(mean_value_losses)), mean_value_losses-std_value_losses, mean_value_losses+std_value_losses, alpha=0.2)
        axs[2].fill_between(range(len(mean_policy_losses)), mean_policy_losses-std_policy_losses, mean_policy_losses+std_policy_losses, alpha=0.2)
        axs[3].fill_between(range(len(mean_total_loss)), mean_total_loss-std_total_loss, mean_total_loss+std_total_loss, alpha=0.2)


    # plot marlisa results as extra line in the reward plot
    # marlisa_reward = read_marlisa_results()
    # if smooth_data:
    #     marlisa_reward = smooth(marlisa_reward, window_size)
    # axs[0].plot(marlisa_reward, color='black', label='MARLISA')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()

    # set common x label
    fig.text(0.5, 0.04, 'time step', ha='center')

    axs[0].set_ylabel('reward')
    axs[1].set_ylabel('value loss')
    axs[2].set_ylabel('policy loss')
    axs[3].set_ylabel('total loss')

    # axs[0].set_ylim(top=1)
    # axs[1].set_ylim(0, 100)
    # axs[2].set_ylim(-300,300)
    
    plt.tight_layout()
    plt.show()
    plt.savefig('results/all_plots.png')


def smooth(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def read_marlisa_results():

    rewards = []
    with open('marlisa2.log') as f:

        lines = f.readlines()

        for line in lines:
            if 'Time step' in line:
                reward_str = line.split('Rewards: ')[1]
                reward_str = reward_str.replace('[', '').replace(']', '')
                reward = np.array([float(reward) for reward in reward_str.split(', ')]).sum()
                rewards.append(reward)

    rewards = np.array(rewards)

    return rewards


def main():

    names = os.listdir('results/')
    names.remove('all_plots.png')

    names = ['SEAC_2023-12-02_22-36-16'] # testing

    smooth_data = True
    window_size = 100
    plot(names, smooth_data, window_size)


if __name__ == "__main__":
    main()
