from stable_baselines3.sac import SAC
from citylearn.citylearn import CityLearnEnv
import torch

from envs import make_vec_envs
from a2c import A2C

dataset_name = 'citylearn_challenge_2022_phase_1'
num_procs = 4
time_limit = 1000
device = "cpu"

envs = make_vec_envs(dataset_name, num_procs, time_limit, device)

# for obs in env.observation_space:
#     print(obs.shape)
# print(env.observation_names[0])
# print(len(env.observation_names[0]))
# env = NormalizedObservationWrapper(env)
# env = StableBaselines3Wrapper(env)
agents = [
        A2C(i, osp, asp)
        for i, (osp, asp) in enumerate(zip(envs.observation_space, envs.action_space))
    ]
print(len(agents))
obs = envs.reset()

print(obs)
print(len(obs))
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


# model.learn(total_timesteps=env.time_steps*2)

# # evaluate
# observations = env.reset()

# while not env.done:
#     actions, _ = model.predict(observations, deterministic=True)
#     observations, _, _, _ = env.step(actions)

# kpis = env.evaluate()
# kpis = kpis.pivot(index='cost_function', columns='name', values='value')
# kpis = kpis.dropna(how='all')
# print(kpis)
