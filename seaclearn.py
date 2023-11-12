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
import pickle

from citylearn.citylearn import CityLearnEnv, EvaluationCondition

config = {
    # Dataset information
    # "dataset_name": "data/citylearn_challenge_2022_phase_1/schema.json",
    "dataset_name": "data/citylearn_challenge_2022_phase_1_normalized_period/schema.json",
    "num_procs": 4,
    "time_limit": 1000,
    "seed": 42,

    # RL params
    "gamma": 0.99,
    "use_gae": False,
    "gae_lambda": 0.95,
    "use_proper_time_limits": True,

    # Training params
    "entropy_coef": 0.05,
    "value_loss_coef": 0.5,
    "seac_coef": 1.0,
    "max_grad_norm": 0.5,
    "device": "cpu",

    # Environment settings
    "num_steps": 5,
    "num_env_steps": 100000,
    
    # Environment wrappers
    "flatten_observation": True,
    "flatten_action": True,

    "recurrent_policy": True
}

# Environment wrappers
wrappers = (
    FlattenObservation,
    FlattenAction,
)

# Initialize agents
def init_agents(envs, obs):

    agents = [
        A2C(i, osp, asp, num_processes=config['num_procs'], recurrent_policy=config['recurrent_policy'])
        for i, (osp, asp) in enumerate(zip(envs.observation_space, envs.action_space))
    ]

    for i in range(len(obs)):
        agents[i].storage.obs[0].copy_(obs[i])
        agents[i].storage.to(config['device'])

    return agents

# Train agents
def train(agents, envs):

    policy_losses, value_losses, dist_entropies, importance_samplings, seac_policy_losses, seac_value_losses, rewards = [], [], [], [], [], [], []

    num_updates = (
        int(config['num_env_steps']) // config['num_steps'] // config['num_procs']
    )
    # print current time
    print('Started training at:', datetime.datetime.now())
    for j in range(1, num_updates + 1):

        for step in range(config['num_steps']):
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

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            bad_masks = torch.FloatTensor(
                [
                    [0.0] if info.get("TimeLimit.truncated", False) else [1.0]
                    for info in infos
                ]
            )
            # Store relevant information
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

        # Compute returns for each agent
        for agent in agents:
            agent.compute_returns(config['use_gae'], config['gamma'], config['gae_lambda'], config['use_proper_time_limits'])

        # TODO: Make this more readable and efficient
        total_policy_loss = 0
        total_value_loss = 0
        total_dist_entropy = 0
        total_importance_sampling = 0
        total_seac_policy_loss = 0
        total_seac_value_loss = 0
        for agent in agents:
            loss = agent.update([a.storage for a in agents], config['value_loss_coef'], config['entropy_coef'], config['seac_coef'], config['max_grad_norm'], config['device'])
            total_policy_loss = loss['policy_loss']
            total_value_loss = loss['value_loss']
            total_dist_entropy = loss['dist_entropy']
            total_importance_sampling = loss['importance_sampling']
            total_seac_policy_loss += loss['seac_policy_loss']
            total_seac_value_loss += loss['seac_value_loss']
        policy_losses.append(total_policy_loss)
        value_losses.append(total_value_loss)
        dist_entropies.append(total_dist_entropy)
        importance_samplings.append(total_importance_sampling)
        seac_policy_losses.append(total_policy_loss)
        seac_value_losses.append(total_value_loss)
        rewards.append(np.array(reward).sum(axis=1).mean())
        
        for agent in agents:
            agent.storage.after_update()

        if j % 1000 == 0:
            print(f'Update {j}/{num_updates}')

    print('Finished training at:', datetime.datetime.now())
    return agents, policy_losses, value_losses, dist_entropies, importance_samplings, seac_policy_losses, seac_value_losses, rewards


# Save agent models
def save_results(agents, policy_losses, value_losses, dist_entropies, importance_samplings, seac_policy_losses, seac_value_losses, rewards, run_nr, name):

    save_dir = f"./results/{name}"
    agents_dir = f"{save_dir}/agents/{run_nr}"
    train_logs_dir = f"{save_dir}/train_logs/{run_nr}"

    for agent in agents:
        save_at = f'{agents_dir}/agent{agent.agent_id}'
        os.makedirs(save_at, exist_ok=True)
        agent.save(save_at)

    # Save training logs
    # TODO: Make this more readable and efficient
    os.makedirs(f'{train_logs_dir}', exist_ok=True)
    np.save(f'{train_logs_dir}/rewards.npy', np.array(rewards))
    np.save(f'{train_logs_dir}/valueloss.npy', np.array(value_losses))
    np.save(f'{train_logs_dir}/policyloss.npy', np.array(policy_losses))
    np.save(f'{train_logs_dir}/distentropy.npy', np.array(dist_entropies))
    np.save(f'{train_logs_dir}/importancesampling.npy', np.array(importance_samplings))
    np.save(f'{train_logs_dir}/seacpolicyloss.npy', np.array(seac_policy_losses))
    np.save(f'{train_logs_dir}/seacvalueloss.npy', np.array(seac_value_losses))

    print(f"Saved agents in {name}")

# Save hyperparameters and other settings
def save_config(config, name):

    save_dir = f"./results/{name}"
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/config.txt', 'w') as f:
        for key, value in config.items():
            f.write(f'{key}: {value}\n')


# Load agent models
def load_agents(envs, name, evaluation = False):

    n = 1
    run_nr = 0
    save_dir = f"./results/{name}/agents/{run_nr}"

    if not evaluation:
        n = config['num_procs']

    agents = []
    for i, (osp, asp) in enumerate(zip(envs.observation_space, envs.action_space)):
        agent = A2C(i, osp, asp, num_processes = n, recurrent_policy = config['recurrent_policy'])
        model_path = f"{save_dir}/agent{i}"
        agent.restore(model_path)
        agents.append(agent)

    return agents


# Evaluate function without vectorized envs
def evaluate_single_env(env, agents, render=False, animation=False):
    
    # Evaluation settings
    evaluation_steps = 1000
    render_freq = 10

    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    
    # Initialize recurrent hidden states and masks for recurrent policies
    n_recurrent_hidden_states = [
        torch.zeros(
            evaluation_steps, agent.model.recurrent_hidden_state_size, device=config['device']
        )
        for agent in agents
    ]
    masks = torch.zeros(evaluation_steps, 1, device=config['device'])
    
    frames = []
    for j in range(evaluation_steps):

        n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(
            *[
                agent.model.act(
                    obs[i],
                    n_recurrent_hidden_states[i],
                    masks[i],
                )
                for i, agent in enumerate(agents)
            ]
        )

        # n_actions = []
        # for i, agent in enumerate(agents):
        #     with torch.no_grad():
        #         n_value, n_action, n_action_log_prob, n_recurrent_hidden_states[i] = agent.model.act(obs[i], n_recurrent_hidden_states[i], masks)
        #         n_actions.append(n_action)

        # n_actions = [tensor.detach().cpu().numpy() for tensor in n_actions]
        obs, rewards, done, info = env.step(n_action)
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

    # Evaluate agents and generate cost function values
    kpis = env.evaluate(baseline_condition=EvaluationCondition.WITHOUT_STORAGE_BUT_WITH_PARTIAL_LOAD_AND_PV)
    kpis = kpis.pivot(index='cost_function', columns='name', values='value')
    kpis = kpis.dropna(how='all')
    print(kpis)

    env.close()


# Evaluate function with vectorized envs (isn't currently used)
def evaluate(agents):

    eval_envs = make_vec_envs(env_name=config['dataset_name'],
                              parallel=config['num_procs'],
                              time_limit=None, # time_limit=time_limit,
                              wrappers=wrappers,
                              device=config['device'],
                              monitor_dir=None,
                              evaluate=True
                              )
    n_obs = eval_envs.reset()

    ep_length = 1000
    n_recurrent_hidden_states = [
        torch.zeros(
            ep_length, agent.model.recurrent_hidden_state_size, device=config['device']
        )
        for agent in agents
    ]

    masks = torch.zeros(ep_length, 1, device=config['device'])

    performed_actions = []

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

    kpis = eval_envs.env_method("evaluate", baseline_condition=EvaluationCondition.WITHOUT_STORAGE_BUT_WITH_PARTIAL_LOAD_AND_PV, indices=0)[0]
    kpis = kpis.pivot(index='cost_function', columns='name', values='value')
    kpis = kpis.dropna(how='all')
    print(kpis)

    eval_envs.close()


def main():

    train_new_agent = False
    nr_runs = 3

    if train_new_agent:

        now = datetime.datetime.now()
        name = "SEAC_" + now.strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Training new agent: {name}")
        save_config(config, name)

        for run_nr in range(nr_runs):
            
            print("Starting run number:", run_nr)

            # Make vectorized envs
            torch.set_num_threads(1)
            envs = make_vec_envs(env_name=config['dataset_name'],
                                parallel=config['num_procs'],
                                time_limit=None, # time_limit=time_limit,
                                wrappers=wrappers,
                                device=config['device'],
                                monitor_dir=None
                                )

            obs = envs.reset()

            # Initialize agents
            agents = init_agents(envs, obs)

            # Train models
            agents, policy_losses, value_losses, dist_entropies, importance_samplings, seac_policy_losses, seac_value_losses, rewards = train(agents, envs)

            # Save trained models
            save_results(agents, policy_losses, value_losses, dist_entropies, importance_samplings, seac_policy_losses, seac_value_losses, rewards, run_nr, name)
            print("Saved agent.")

            envs.close()

    else:

        name = "SEAC_2023-11-10_12-54-45" # name of the model to load

        env = make_env(env_name = config['dataset_name'],
                       rank = 1,
                       time_limit=None,
                       wrappers = [],
                       monitor_dir = None
                       )

        print("Loading agents...")
        agents = load_agents(env, name, evaluation = True)
        print("Agents loaded!")

        print("Evaluating...")
        evaluate_single_env(env, agents)


if __name__ == '__main__':
    main()
