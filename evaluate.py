from envs import make_env
from wrappers import FlattenObservation, FlattenAction, DiscreteActionWrapperFix
from citylearn.wrappers import NormalizedObservationWrapper, DiscreteActionWrapper
from a2c import A2C
import torch
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from typing import List, Mapping, Tuple
from citylearn.citylearn import EvaluationCondition


def read_config(name):

    config = {}
    with open(f'results/{name}/config.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(": ")
            key, value = line[0], line[1].strip()
            if value == 'True':
                config[key] = True
            elif value == 'False':
                config[key] = False
            elif value.isnumeric():
                config[key] = int(value)
            else:
                try:
                    config[key] = float(value)
                except ValueError:
                    config[key] = value

    return config


def get_wrappers(config):
    
    wrappers = []
    if config['normalize_observations']:
        wrappers.append(NormalizedObservationWrapper)

    if config['discrete_policy']:
        wrappers.append(DiscreteActionWrapper)

    return wrappers


# Load agent models
def load_agents(envs, name, config):

    run_nr = 0
    save_dir = f"./results/{name}/agents/{run_nr}"

    agents = []
    for i, (osp, asp) in enumerate(zip(envs.observation_space, envs.action_space)):
        agent = A2C(i, osp, asp, hidden_size=config['hidden_size'], num_processes=config['num_procs'], recurrent_policy=config['recurrent_policy'], discrete_policy=config['discrete_policy'], default_bin_size=config['default_bin_size'])
        model_path = f"{save_dir}/agent{i}"
        agent.restore(model_path)
        agents.append(agent)

    return agents


# Evaluate function without vectorized envs
def evaluate_single_env(env, agents, config, render=False, animation=False):
    
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
    
    action_list = []
    frames = []
    for j in range(evaluation_steps):
        
        n_actions = []
        for i, agent in enumerate(agents):
            with torch.no_grad():
                n_value, n_action, n_action_log_prob, n_recurrent_hidden_states[i] = agent.model.act(obs[i], n_recurrent_hidden_states[i], masks, deterministic=True)
                n_actions.append(n_action)

        n_actions = [tensor.detach().cpu().numpy() for tensor in n_actions]
        action_list.append(n_actions)

        obs, rewards, done, info = env.step(n_actions)
        obs = torch.tensor(obs, dtype=torch.float32)

        # print("Actions: ", n_actions)
        # elec_storage_index = env.observation_names[0].index('electrical_storage_soc')
        # for ob in obs:
        #     print('Battery level:', ob[elec_storage_index])

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

    return action_list


def plot_actions(actions_list: List[List[float]], title: str, env, config) -> plt.Figure:
    """Plots action time series for different buildings

    Parameters
    ----------
    actions_list: List[List[float]]
        List of actions where each element with index, i,
        in list is a list of the actions for different buildings
        taken at time step i.
    title: str
        Plot axes title

    Returns
    -------
    fig: plt.Figure
        Figure with plotted axes

    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 2))
    columns = [b.name.replace('_', ' ') for b in env.buildings]
    plot_data = pd.DataFrame(actions_list, columns=columns)
    x = list(range(plot_data.shape[0]))

    for c in plot_data.columns:
        y = plot_data[c].tolist()
        ax.plot(x, y, label=c)

    if config['discrete_policy']:
        ax.set_ylim(-0.1, config['default_bin_size']-0.9)
        ax.set_yticks([i for i in range(config['default_bin_size'])])
    else:
        ax.set_ylim(-1.1, 1.1)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0)
    ax.set_xlabel('Time step')
    ax.set_ylabel(r'Action $a_t$')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
    ax.set_title(title)

    return fig


# def plot_building_kpis(envs: Mapping[str, CityLearnEnv]) -> plt.Figure:
#     """Plots electricity consumption, cost and carbon emissions
#     at the building-level for different control agents in bar charts.

#     Parameters
#     ----------
#     envs: Mapping[str, CityLearnEnv]
#         Mapping of user-defined control agent names to environments
#         the agents have been used to control.

#     Returns
#     -------
#     fig: plt.Figure
#         Figure containing plotted axes.
#     """

#     kpis_list = []

#     for k, v in envs.items():
#         kpis = get_kpis(v)
#         kpis = kpis[kpis['level']=='building'].copy()
#         kpis['building_id'] = kpis['name'].str.split('_', expand=True)[1]
#         kpis['building_id'] = kpis['building_id'].astype(int).astype(str)
#         kpis['env_id'] = k
#         kpis_list.append(kpis)

#     kpis = pd.concat(kpis_list, ignore_index=True, sort=False)
#     kpi_names= kpis['kpi'].unique()
#     column_count_limit = 3
#     row_count = math.ceil(len(kpi_names)/column_count_limit)
#     column_count = min(column_count_limit, len(kpi_names))
#     building_count = len(kpis['name'].unique())
#     env_count = len(envs)
#     figsize = (3.0*column_count, 0.3*env_count*building_count*row_count)
#     fig, _ = plt.subplots(
#         row_count, column_count, figsize=figsize, sharey=True
#     )

#     for i, (ax, (k, k_data)) in enumerate(zip(fig.axes, kpis.groupby('kpi'))):
#         sns.barplot(x='value', y='name', data=k_data, hue='env_id', ax=ax)
#         ax.axvline(1.0, color='black', linestyle='--', label='Baseline')
#         ax.set_xlabel(None)
#         ax.set_ylabel(None)
#         ax.set_title(k)

#         if i == len(kpi_names) - 1:
#             ax.legend(
#                 loc='upper left', bbox_to_anchor=(1.3, 1.0), framealpha=0.0
#             )
#         else:
#             ax.legend().set_visible(False)

#         for s in ['right','top']:
#             ax.spines[s].set_visible(False)

#         for p in ax.patches:
#             ax.text(
#                 p.get_x() + p.get_width(),
#                 p.get_y() + p.get_height()/2.0,
#                 p.get_width(), ha='left', va='center'
#             )

#     plt.tight_layout()
#     return fig


# def plot_simulation_summary(envs: Mapping[str, CityLearnEnv]):
#     """Plots KPIs, load and battery SoC profiles for different control agents.

#     Parameters
#     ----------
#     envs: Mapping[str, CityLearnEnv]
#         Mapping of user-defined control agent names to environments
#         the agents have been used to control.
#     """

#     _ = plot_building_kpis(envs)
#     print('Building-level KPIs:')
#     plt.show()
#     _ = plot_building_load_profiles(envs)
#     print('Building-level load profiles:')
#     plt.show()
#     _ = plot_battery_soc_profiles(envs)
#     print('Battery SoC profiles:')
#     plt.show()
#     _ = plot_district_kpis(envs)
#     print('District-level KPIs:')
#     plt.show()
#     print('District-level load profiles:')
#     _ = plot_district_load_profiles(envs)
#     plt.show()


def main():

    name = "3DiscNoRec5mil512Hidden" # name of the model to load
    render = False
    animation = False

    config = read_config(name)

    wrappers = get_wrappers(config)

    env = make_env(env_name = config['dataset_name'],
                    rank = 1,
                    time_limit=None,
                    wrappers = wrappers,
                    default_bin_size=config['default_bin_size'],
                    monitor_dir = None
                    )

    print("Loading agents...")
    agents = load_agents(env, name, config)
    print("Agents loaded!")

    print("Evaluating...")
    action_list = evaluate_single_env(env, agents, config, render=render, animation=animation)

    print("Plotting actions...")
    if config['discrete_policy']:
        title = 'Discrete Actions'
    else:
        title = 'Continuous Actions'
    fig = plot_actions(action_list[:97], title, env, config)
    plt.savefig(f'results/{name}/plots/actions.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()