from wrappers import FlattenObservation, FlattenAction, DiscreteActionWrapperFix
from citylearn.wrappers import NormalizedObservationWrapper, DiscreteActionWrapper
import torch
import os
import datetime
import json

from typing import List, Mapping, Tuple

from citylearn.citylearn import CityLearnEnv
from envs import make_env, make_vec_envs
from a2c import A2C

from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd

from citylearn.citylearn import EvaluationCondition
from citylearn.data import DataSet

config = {
    # Dataset information
    "dataset_name": "data/citylearn_challenge_2022_phase_1/schema.json",
    # "dataset_name": "data/citylearn_challenge_2022_phase_1_normalized_period/schema.json",
    "num_procs": 4,
    "seed": 42,

    # RL params
    "hidden_size" : 256,
    "gamma": 0.99,
    "use_gae": False,
    "gae_lambda": 0.95,
    "use_proper_time_limits": True,
    'random_warmup_steps': 0,

    # Training params
    "entropy_coef": 0.1,
    "value_loss_coef": 0.1,
    "seac_coef": 1.0,
    "max_grad_norm": 0.5,
    "device": "cpu",

    # Environment settings
    "num_steps": 5,
    "num_env_steps": 100000,
    
    "recurrent_policy": False,
    "discrete_policy": False,
    "default_bin_size": 3, # only used if discrete_policy is True

    'normalize_observations': True
}

# Environment wrappers
wrappers = []

if config['normalize_observations']:
    wrappers.append(NormalizedObservationWrapper)

if config['discrete_policy']:
    wrappers.append(DiscreteActionWrapperFix)
elif config['discrete_policy']:
    wrappers.append(DiscreteActionWrapper)
else:
    wrappers.append(FlattenAction)

wrappers.append(FlattenObservation)

def set_active_observations(
    schema: dict, active_observations: List[str]
) -> dict:
    """Set the observations that will be part of the environment's
    observation space that is provided to the control agent.

    Parameters
    ----------
    schema: dict
        CityLearn dataset mapping used to construct environment.
    active_observations: List[str]
        Names of observations to set active to be passed to control agent.

    Returns
    -------
    schema: dict
        CityLearn dataset mapping with active observations set.
    """

    active_count = 0

    for o in schema['observations']:
        if o in active_observations:
            schema['observations'][o]['active'] = True
            active_count += 1
        else:
            schema['observations'][o]['active'] = False

    valid_observations = list(schema['observations'].keys())
    assert active_count == len(active_observations),\
        'the provided observations are not all valid observations.'\
          f' Valid observations in CityLearn are: {valid_observations}'

    return schema

# Initialize agents
def init_agents(envs, obs):

    agents = [
        A2C(i, osp, asp, hidden_size=config['hidden_size'], num_processes=config['num_procs'], num_steps=config['num_steps'], recurrent_policy=config['recurrent_policy'], discrete_policy=config['discrete_policy'], default_bin_size=config['default_bin_size'])
        for i, (osp, asp) in enumerate(zip(envs.observation_space, envs.action_space))
    ]

    for i in range(len(obs)):
        agents[i].storage.obs[0].copy_(obs[i])
        agents[i].storage.to(config['device'])

    return agents

# Train agents
def train(agents, envs):

    policy_losses, value_losses, dist_entropies, importance_samplings, seac_policy_losses, seac_value_losses, rewards = [], [], [], [], [], [], []

    n_recurrent_hidden_states = ([
        torch.zeros(
            1, agent.model.recurrent_hidden_state_size, device=config['device']
        )
        for agent in agents
    ])

    num_updates = (
        int(config['num_env_steps']) // config['num_steps'] // config['num_procs']
    )
    # print current time
    print('Started training at:', datetime.datetime.now())
    for j in range(1, num_updates + 1):

        for step in range(config['num_steps']):

            if 1 < j < config['random_warmup_steps']:

                # Sample random actions
                with torch.no_grad():
                    n_action = torch.tensor(np.array([[a.sample() for _ in range(config['num_procs'])] for a in envs.action_space]))                        
                obs, reward, done, infos = envs.step(n_action)

            else:
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
            total_policy_loss += loss['policy_loss']
            total_value_loss += loss['value_loss']
            total_dist_entropy += loss['dist_entropy']
            total_importance_sampling += loss['importance_sampling']
            total_seac_policy_loss += loss['seac_policy_loss']
            total_seac_value_loss += loss['seac_value_loss']
        policy_losses.append(total_policy_loss)
        value_losses.append(total_value_loss)
        dist_entropies.append(total_dist_entropy)
        importance_samplings.append(total_importance_sampling)
        seac_policy_losses.append(total_policy_loss)
        seac_value_losses.append(total_value_loss)
        rewards.append(np.array(reward).sum(axis=1).mean())

        # if j & 100 == 0:
        #     print("Mean reward: ", np.array(reward).sum(axis=1).mean())
        #     print("Policy loss: ", total_policy_loss)
        #     print("Value loss: ", total_value_loss)
        #     print("Total loss?: ", total_policy_loss + total_value_loss - total_dist_entropy + total_seac_policy_loss + total_seac_value_loss)
        
        for agent in agents:
            agent.storage.after_update()

        if j % 1 == 0:
            print(f'Update {j}/{num_updates}')

    print('Finished training at:', datetime.datetime.now())
    return agents, policy_losses, value_losses, dist_entropies, importance_samplings, seac_policy_losses, seac_value_losses, rewards

# Save agent models
def save_results(agents, policy_losses, value_losses, dist_entropies, importance_samplings, seac_policy_losses, seac_value_losses, rewards, run_nr, name):

    save_dir = f"./results/{name}"
    agents_dir = f"{save_dir}/agents/{run_nr}"
    train_logs_dir = f"{save_dir}/train_logs/{run_nr}"

    print(train_logs_dir)

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
        agent = A2C(i, osp, asp, hidden_size=config['hidden_size'], num_processes=config['num_procs'], recurrent_policy=config['recurrent_policy'], discrete_policy=config['discrete_policy'], default_bin_size=config['default_bin_size'])
        model_path = f"{save_dir}/agent{i}"
        agent.restore(model_path)
        agents.append(agent)

    return agents

def main():

    nr_runs = 1

    now = datetime.datetime.now()
    name = "SEAC_" + now.strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Training new agent: {name}")
    save_config(config, name)

    f = open('data/citylearn_challenge_2022_phase_1/schema.json')
    schema = json.load(f)
    schema['root_directory'] = '/home/wortel/Documents/seaclearn/data/citylearn_challenge_2022_phase_1'

    active_observations = ["month",
                           "day_type",
                           "hour",
                           "solar_generation",
                           "electrical_storage_soc",
                           "net_electricity_consumption",
                           "electricity_pricing"]

    schema = set_active_observations(schema, active_observations)

    for run_nr in range(nr_runs):
        
        print("Starting run number:", run_nr)

        # Make vectorized envs
        torch.set_num_threads(1)
        envs = make_vec_envs(env_name=schema,
                            parallel=config['num_procs'],
                            time_limit=None, # time_limit=time_limit,
                            wrappers=wrappers,
                            default_bin_size=config['default_bin_size'],
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

if __name__ == '__main__':
    main()
