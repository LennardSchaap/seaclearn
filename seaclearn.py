from stable_baselines3.sac import SAC
from citylearn.citylearn import CityLearnEnv
from wrappers import RecordEpisodeStatistics, SquashDones
import torch

from envs import make_vec_envs
from a2c import A2C


def main():

    dataset_name = 'citylearn_challenge_2022_phase_1'
    num_procs = 2
    time_limit = 1000
    device = "cpu"
    seed = 42

    num_steps = 5
    num_env_steps = 10000
    
    wrappers = (
            RecordEpisodeStatistics,
            SquashDones,
        )

    torch.set_num_threads(1)
    envs = make_vec_envs(dataset_name, seed, num_procs, time_limit, wrappers, device, monitor_dir=None)
    
    agents = [
            A2C(i, osp, asp, num_processes=num_procs)
            for i, (osp, asp) in enumerate(zip(envs.observation_space, envs.action_space))
        ]

    obs = envs.reset()

    # observation = torch.stack(obs).moveaxis(0,1)
    # test = list(observation)
    # obs = test

    # print(obs)
    # print(len(obs))

    # TODO: Multithreading werkt niet???, daarom is obs niet verdeeld over 4 processen en is het een 4x5x28

    for i in range(len(obs)):
        agents[i].storage.obs[0].copy_(obs[i])
        agents[i].storage.to(device)

    # for step in range(10):
    #     with torch.no_grad():
    #         n_value, n_action, n_recurrent_hidden_states = zip(
    #             *[
    #                 agent.model.act(
    #                     agent.storage.obs[step],
    #                     agent.storage.recurrent_hidden_states[step],
    #                     agent.storage.masks[step],
    #                 )
    #                 for agent in agents
    #             ]
    #         )
    #         print(n_action)
    #     # Obser reward and next obs
    #     obs, reward, done, infos = envs.step(n_action)


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
            obs, reward, done, infos = envs.step(n_action)
            # envs.envs[0].render()

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            bad_masks = torch.FloatTensor(
                [
                    [0.0] if info.get("TimeLimit.truncated", False) else [1.0]
                    for info in infos
                ]
            )
            for i in range(len(agents)):
                agents[i].storage.insert(
                    obs[i],
                    n_recurrent_hidden_states[i],
                    n_action[i],
                    None,
                    n_value[i],
                    reward[:, i].unsqueeze(1),
                    masks,
                    bad_masks,
                )

        # value_loss, action_loss, dist_entropy = agent.update(rollouts)
        for agent in agents:
            agent.compute_returns()

        for agent in agents:
            loss = agent.update([a.storage for a in agents])

        for agent in agents:
            agent.storage.after_update()

        envs.close()

if __name__ == '__main__':
    main()