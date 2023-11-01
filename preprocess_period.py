import numpy as np
import pandas as pd
import os
import shutil

# save all dataframes as csv files in a new folder called citylearn_challenge_2022_phase_1_normalized

# create new folder
if not os.path.exists('data/citylearn_challenge_2022_phase_1_normalized_period'):
    os.makedirs('data/citylearn_challenge_2022_phase_1_normalized_period')

# load buildings
buildings = []
for i in range(1, 6):
    buildings.append(pd.read_csv('data/citylearn_challenge_2022_phase_1/building_' + str(i) + '.csv'))

# # apply periodic transformation to Month, Hour, Day Type features
# for building in buildings:
#     building.insert(0, 'Month_cos', np.cos(2 * np.pi * building['Month'] / 12))
#     building.insert(1, 'Month_sin', np.sin(2 * np.pi * building['Month'] / 12))
#     building.insert(2, 'Hour_cos', np.cos(2 * np.pi * building['Hour'] / 24))
#     building.insert(3, 'Hour_sin', np.sin(2 * np.pi * building['Hour'] / 24))
#     building.insert(4, 'Day Type_cos', np.cos(2 * np.pi * building['Day Type'] / 8))
#     building.insert(5, 'Day Type_sin', np.sin(2 * np.pi * building['Day Type'] / 8))

for building in buildings:
    building['Month'] = np.sin(2 * np.pi * building['Month'] / 12)
    building['Hour'] = np.sin(2 * np.pi * building['Hour'] / 24)
    building['Day Type'] = np.sin(2 * np.pi * building['Day Type'] / 8)

# # remove original Month, Hour, Day Type features
# for building in buildings:
#     building.drop(columns=['Month', 'Hour', 'Day Type'], inplace=True)

# save buildings
for i in range(5):
    buildings[i].to_csv('data/citylearn_challenge_2022_phase_1_normalized_period/building_' + str(i + 1) + '.csv', index=False)

# copy remaining files to new folder
shutil.copy('data/citylearn_challenge_2022_phase_1/carbon_intensity.csv', 'data/citylearn_challenge_2022_phase_1_normalized_period/carbon_intensity.csv')
shutil.copy('data/citylearn_challenge_2022_phase_1/weather.csv', 'data/citylearn_challenge_2022_phase_1_normalized_period/weather.csv')
shutil.copy('data/citylearn_challenge_2022_phase_1/pricing.csv', 'data/citylearn_challenge_2022_phase_1_normalized_period/pricing.csv')
shutil.copy('data/citylearn_challenge_2022_phase_1/schema.json', 'data/citylearn_challenge_2022_phase_1_normalized_period/schema.json')