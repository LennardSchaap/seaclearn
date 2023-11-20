from wrappers import RecordEpisodeStatistics, SquashDones, FlattenObservation, FlattenAction
from citylearn.wrappers import NormalizedObservationWrapper
import torch
import os
import datetime
import sys

from envs import make_env, make_vec_envs
from a2c import A2C

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation
import numpy as np
import uuid
from os import path

from citylearn.citylearn import CityLearnEnv, EvaluationCondition

# Dataset information

dataset_name = './data/citylearn_challenge_2022_phase_1/schema.json'
# dataset_name = 'citylearn_challenge_2022_phase_1'
num_procs = 4
time_limit = 1000
seed = 42

# RL params
gamma = 0.99
use_gae = False
gae_lambda = 0.95
use_proper_time_limits = True

# Training params
entropy_coef = 0.01
value_loss_coef = 0.5
seac_coef = 1.0
max_grad_norm = 0.5
device = "cpu"
# variance = 0.5

# Environment settings
num_steps = 5
num_env_steps = 50000

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

    # # use global variance
    # global variance

    policy_losses, value_losses, rewards = [], [], []

    num_updates = (
        int(num_env_steps) // num_steps // num_procs
    )
    # print current time
    print('Started at:', datetime.datetime.now())
    for j in range(1, num_updates + 1):

        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(
                    *[
                        agent.model.act(
                            agent.storage.obs[step],
                            agent.storage.recurrent_hidden_states[step],
                            agent.storage.masks[step],
                        )
                        for agent in agents
                    ]
                )
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
                    n_action_log_prob[i],
                    n_value[i],
                    reward[:, i].unsqueeze(1),
                    masks,
                    bad_masks,
                )


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

        if j % 1000 == 0:
            print(f'update {j}')
        #     print('variance', variance)

        # variance *= 0.999999
        # variance = max(0.01, variance)

    print('Finished at:', datetime.datetime.now())
    return agents, policy_losses, value_losses, rewards


# Save agent models
def save_agents(agents):

    now = datetime.datetime.now()
    name = now.strftime("%Y-%m-%d_%H-%M-%S")

    print(f"Saved agents in {name}")

    save_dir = f"./results/trained_models/{name}"

    for agent in agents:
        save_at = path.join(save_dir, f"agent{agent.agent_id}")
        os.makedirs(save_at, exist_ok=True)
        agent.save(save_at)

# Load agent models
def load_agents(envs, agent_dir, evaluation = False):
    agents = []
    n = 1
    save_dir = "./results/trained_models/"

    if not evaluation:
        n = num_procs

    for i, (osp, asp) in enumerate(zip(envs.observation_space, envs.action_space)):
        agent = A2C(i, osp, asp, num_processes = n)
        model_path = f"{save_dir}{agent_dir}/agent{i}"
        agent.restore(model_path)
        agents.append(agent)

    return agents

def evaluate_single_env(env, agents, render=False, animation=False):

    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    evaluation_steps = 1000
    render_freq = 10

    n_recurrent_hidden_states = [
        torch.zeros(
            evaluation_steps, agent.model.recurrent_hidden_state_size, device=device
        )
        for agent in agents
    ]
    masks = torch.zeros(evaluation_steps, 1, device=device)
    frames = []

    for j in range(evaluation_steps):
        n_actions = []  
        for i, agent in enumerate(agents):
            with torch.no_grad():
                n_value, n_action, n_action_log_prob, n_recurrent_hidden_states[i] = agent.model.act(obs[i], n_recurrent_hidden_states[i], masks)
                n_actions.append(n_action)
        n_actions = [tensor.detach().cpu().numpy() for tensor in n_actions]
        obs, rewards, done, info = env.step(n_actions)
        obs = torch.tensor(obs, dtype=torch.float32)

        if render and not j % render_freq:
            frame_data = env.render()
            frames.append(frame_data)

    print("Done evaluating.")

    if render:
        for frame_data in frames:
            plt.imshow(frame_data)
            plt.pause(0.01)
            plt.draw()

        plt.show()

    if animation:
        print("Creating animation...")
        fig, ax = plt.subplots()
        ax.axis('off')
        anim = FuncAnimation(fig, lambda x: plt.imshow(frames[x], animated=True), frames=len(frames), blit=False)
        anim.save("evaluation.mp4", writer="ffmpeg")
        plt.show()
        print("Animation created.")

    kpis = env.evaluate(baseline_condition=EvaluationCondition.WITHOUT_STORAGE_BUT_WITH_PARTIAL_LOAD_AND_PV)
    kpis = kpis.pivot(index='cost_function', columns='name', values='value')
    kpis = kpis.dropna(how='all')
    print(kpis)

# Evaluate agents
def evaluate(agents):

    eval_envs = make_vec_envs(env_name=dataset_name,
                              parallel=num_procs,
                              time_limit=None, # time_limit=time_limit,
                              wrappers=wrappers,
                              device=device,
                              monitor_dir=None
                              )
    n_obs = eval_envs.reset()

    ep_length = 7000
    n_recurrent_hidden_states = [
        torch.zeros(
            ep_length, agent.model.recurrent_hidden_state_size, device=device
        )
        for agent in agents
    ]

    masks = torch.zeros(ep_length, 1, device=device)

    performed_actions = []

    done = np.array([False for _ in range(num_procs)])
    for _ in range(ep_length):
        with torch.no_grad():
            n_value, n_action, n_recurrent_hidden_states = zip(
                *[
                    agent.model.act(
                        n_obs[agent.agent_id], recurrent_hidden_states, masks, 0.000001
                    )
                    for agent, recurrent_hidden_states in zip(
                        agents, n_recurrent_hidden_states
                    )
                ]
            )
        n_obs, rewards, done, infos = eval_envs.step(n_action)

        actions = []
        for tensor in n_action:
            action = tensor.detach().cpu().numpy()[0]
            actions.append(action)
        performed_actions.append(actions)

    #TODO: Correctly evaluate vectorized envs...
    kpis = eval_envs.env_method("evaluate", baseline_condition=EvaluationCondition.WITHOUT_STORAGE_BUT_WITH_PARTIAL_LOAD_AND_PV, indices=0)[0]
    kpis = kpis.pivot(index='cost_function', columns='name', values='value')
    kpis = kpis.dropna(how='all')
    print(kpis)

    eval_envs.close()

def main():

    agent_dir = "2023-11-01_20-04-38"

    train_new_agent = True

    if train_new_agent:

        # Make vectorized envs
        torch.set_num_threads(1)
        envs = make_vec_envs(env_name=dataset_name,
                             parallel=num_procs,
                             time_limit=None, # time_limit=time_limit,
                             wrappers=wrappers,
                             device=device,
                             monitor_dir=None
                             )

        obs = envs.reset()

        # Initialize agents
        agents = init_agents(envs, obs)

        # Train models
        agents, policy_losses, value_losses, rewards = train(agents, envs)

        # Save trained models
        save_agents(agents)
        print("Saved agent.")

        envs.close()

        # Save results
        value_losses = np.array(value_losses)
        rewards = np.array(rewards)
        exp_name = 'experiment_var05_10mil_0999999'
        np.save(f'valueloss_{exp_name}.npy', value_losses)
        np.save(f'rewards_{exp_name}.npy', rewards)

    else:
        env = make_env(env_name = dataset_name,
               rank = 1,
               time_limit=None,
               wrappers = [],
               monitor_dir = None)

        print("Loading agents...")
        agents = load_agents(env, agent_dir, evaluation = True)
        print("Agents loaded!")

        print("Evaluating...")

        evaluate_single_env(env, agents)

if __name__ == '__main__':
    main()
