from stable_baselines3.sac import SAC
from citylearn.citylearn import CityLearnEnv
from wrappers import RecordEpisodeStatistics, SquashDones
import torch

from envs import make_vec_envs
from a2c import A2C

dataset_name = 'citylearn_challenge_2022_phase_1'
num_procs = 1
time_limit = 1000
device = "cpu"
seed = 42
wrappers = (
        RecordEpisodeStatistics,
        SquashDones,
    )

torch.set_num_threads(1)
envs = make_vec_envs(dataset_name, seed, num_procs, time_limit, wrappers, device, monitor_dir=None)

agents = [
        A2C(i, osp, asp)
        for i, (osp, asp) in enumerate(zip(envs.observation_space, envs.action_space))
    ]

obs = envs.reset()

# observation = torch.stack(obs).moveaxis(0,1)
# test = list(observation)
# obs = test

print(len(obs))
print(obs)

# TODO: Multithreading werkt niet???, daarom is obs niet verdeeld over 4 processen en is het een 4x5x28

print("thread")

for i in range(len(obs)):
    agents[i].storage.obs[0].copy_(obs[i])
    agents[i].storage.to(device)

for step in range(10):
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
        print(n_action)
    # Obser reward and next obs
    obs, reward, done, infos = envs.step(n_action)

