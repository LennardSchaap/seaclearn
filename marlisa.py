from citylearn.agents.marlisa import MARLISA as RLAgent
from citylearn.citylearn import CityLearnEnv, EvaluationCondition

dataset_name = 'citylearn_challenge_2022_phase_1'
env = CityLearnEnv(dataset_name, central_agent=False, simulation_end_time_step=1000)
model = RLAgent(env)
model.learn(episodes=2, deterministic_finish=True)

kpis = model.env.evaluate(baseline_condition=EvaluationCondition.WITHOUT_STORAGE_BUT_WITH_PARTIAL_LOAD_AND_PV)
kpis = kpis.pivot(index='cost_function', columns='name', values='value')
kpis = kpis.dropna(how='all')
print(kpis)
