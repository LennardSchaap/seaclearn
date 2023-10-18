
from wrappers import RecordEpisodeStatistics, SquashDones, FlattenObservation, FlattenAction
import torch

from envs import make_vec_envs
from a2c import A2C

import matplotlib.pyplot as plt
import numpy as np

def main():

    dataset_name = 'citylearn_challenge_2022_phase_1'
    num_procs = 2
    time_limit = 1000
    seed = 42

    gamma = 0.99
    use_gae = False
    gae_lambda = 0.95
    use_proper_time_limits = True

    value_loss_coef = 0.5
    entropy_coef = 0.01
    seac_coef = 1.0
    max_grad_norm = 0.5
    device = "cpu"
    
    num_steps = 5
    num_env_steps = 10000000
    
    wrappers = (
            FlattenObservation,
            FlattenAction,
            # RecordEpisodeStatistics,
            # SquashDones,
        )

    torch.set_num_threads(1)
    envs = make_vec_envs(dataset_name, seed, num_procs, time_limit, wrappers, device, monitor_dir=None)
    
    agents = [
            A2C(i, osp, asp, num_processes=num_procs)
            for i, (osp, asp) in enumerate(zip(envs.observation_space, envs.action_space))
        ]

    obs = envs.reset()
    policy_losses, value_losses = [], []

    # observation = torch.stack(obs).moveaxis(0,1)
    # test = list(observation)
    # obs = test

    # print(obs)
    # print(len(obs))

    for i in range(len(obs)):
        agents[i].storage.obs[0].copy_(obs[i])
        agents[i].storage.to(device)

    num_updates = (
        int(num_env_steps) // num_steps // num_procs
    )

    for j in range(1, num_updates + 1):

        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                n_value, n_action, n_recurrent_hidden_states = zip(
                    *[
                        agent.model.act(
                            agent.storage.obs[step],
                            agent.storage.recurrent_hidden_states[step],
                            agent.storage.masks[step],
                        )
                        for agent in agents
                    ]
                )

            # Obser reward and next obs
            print(n_action)

            obs, reward, done, infos = envs.step(n_action)
            return
            # envs.envs[0].render()

            # If done then clean the history of observations.
            # masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            # bad_masks = torch.FloatTensor(
            #     [
            #         [0.0] if info.get("TimeLimit.truncated", False) else [1.0]
            #         for info in infos
            #     ]
            # )
            for i in range(len(agents)):
                agents[i].storage.insert(
                    obs[i],
                    n_recurrent_hidden_states[i],
                    n_action[i],
                    # None,
                    n_value[i],
                    reward[:, i].unsqueeze(1),
                    # masks,
                    # bad_masks,
                )

        # value_loss, action_loss, dist_entropy = agent.update(rollouts)
        for agent in agents:
            agent.compute_returns(use_gae, gamma, gae_lambda, use_proper_time_limits)

        total_policy_loss, total_value_loss = 0, 0
        for agent in agents:
            loss = agent.update([a.storage for a in agents], value_loss_coef, entropy_coef, seac_coef, max_grad_norm, device)
            total_policy_loss += loss['seac_policy_loss']
            total_value_loss += loss['seac_value_loss']
        policy_losses.append(total_policy_loss)
        value_losses.append(total_value_loss)

        for agent in agents:
            agent.storage.after_update()

        if j % 10 == 0:
            print(f'update {j}')

    envs.close()
   
    value_losses = np.array(value_losses)
    np.save('valueloss_testrun.npy', value_losses)

if __name__ == '__main__':
    main()
