from wrappers import RecordEpisodeStatistics, SquashDones, FlattenObservation, FlattenAction
import torch
import os
import datetime

from envs import make_vec_envs
from a2c import A2C

import matplotlib.pyplot as plt
import numpy as np
import uuid
from os import path

from citylearn.citylearn import CityLearnEnv, EvaluationCondition

# Dataset information
dataset_name = 'citylearn_challenge_2022_phase_1'
num_procs = 4
time_limit = 1000
seed = 42

# RL params
gamma = 0.99
use_gae = False
gae_lambda = 0.95
use_proper_time_limits = True

# Training params
value_loss_coef = 0.5
entropy_coef = 0.01
seac_coef = 1.0
max_grad_norm = 0.5
device = "cpu"
variance = 0.5

# Environment settings
num_steps = 5
num_env_steps = 1000000

# Environment wrappers
wrappers = (
    FlattenObservation,
    FlattenAction,
)

# Initialize agents
def init_agents(envs, obs):

    agents = [
        A2C(i, osp, asp, num_processes=num_procs)
        for i, (osp, asp) in enumerate(zip(envs.observation_space, envs.action_space))
    ]

    for i in range(len(obs)):
        agents[i].storage.obs[0].copy_(obs[i])
        agents[i].storage.to(device)

    return agents

# Train agents
def train(agents, envs):

    policy_losses, value_losses, rewards = [], [], []

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
                            variance
                        )
                        for agent in agents
                    ]
                )

            obs, reward, done, infos = envs.step(n_action)
            
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
        rewards.append(np.array(reward).sum(axis=1).mean())
        
        for agent in agents:
            agent.storage.after_update()

        if j % 10 == 0:
            print(f'update {j}')

    return agents, policy_losses, value_losses, rewards

# Save agent models
def save_agents(agents):

    now = datetime.datetime.now()
    name = now.strftime("%Y-%m-%d_%H-%M-%S")

    save_dir = f"./results/trained_models/{name}"

    for agent in agents:
        save_at = path.join(save_dir, f"agent{agent.agent_id}")
        os.makedirs(save_at, exist_ok=True)
        agent.save(save_at)

# Load agent models
def load_agents(envs, agent_dir):
    agents = []

    save_dir = "./results/trained_models/"
    # TODO: Get num agents
    for i, (osp, asp) in enumerate(zip(envs.observation_space, envs.action_space)):
        agent = A2C(i, osp, asp, num_processes=num_procs)
        model_path = f"{save_dir}{agent_dir}/agent{i}"  # Update with the actual path
        agent.restore(model_path)
        agents.append(agent)

    return agents

# Evaluate agents
def evaluate(agents):

    eval_envs = make_vec_envs(dataset_name, seed, num_procs, time_limit, wrappers, device, monitor_dir=None)
    n_obs = eval_envs.reset()

    evaluation_eps = 1000

    n_recurrent_hidden_states = [
        torch.zeros(
            evaluation_eps, agent.model.recurrent_hidden_state_size, device=device
        )
        for agent in agents
    ]
    masks = torch.zeros(evaluation_eps, 1, device=device)

    for _ in range(evaluation_eps):
        with torch.no_grad():
            _, n_action, n_recurrent_hidden_states = zip(
                *[
                    agent.model.act(
                        n_obs[agent.agent_id], recurrent_hidden_states, masks
                    )
                    for agent, recurrent_hidden_states in zip(
                        agents, n_recurrent_hidden_states
                    )
                ]
            )
        
        n_obs, rewards, done, infos = eval_envs.step(n_action)

    #TODO: Correctly evaluate vectorized envs...
    kpis = eval_envs.env_method("evaluate", baseline_condition=EvaluationCondition.WITHOUT_STORAGE_BUT_WITH_PARTIAL_LOAD_AND_PV, indices=0)[0]
    kpis = kpis.pivot(index='cost_function', columns='name', values='value')
    kpis = kpis.dropna(how='all')
    print(kpis)

    eval_envs.close()


def main():

    # Make vectorized envs
    torch.set_num_threads(1)
    envs = make_vec_envs(dataset_name, seed, num_procs, time_limit, wrappers, device, monitor_dir=None)
    obs = envs.reset()

    # Initialize agents
    agents = init_agents(envs, obs)

    # Train models
    agents, policy_losses, value_losses, rewards = train(agents, envs)

    # Save trained models
    save_agents(agents)

    envs.close()

    # Evaluate agents
    # agent_dir = "2023-10-20_14-21-24"
    # agents = load_agents(envs, agent_dir)
    # evaluate(agents)
   

    value_losses = np.array(value_losses)
    rewards = np.array(rewards)
    np.save('valueloss_test_experiment.npy', value_losses)
    np.save('rewards_test_experiment.npy', rewards)

if __name__ == '__main__':
    main()
