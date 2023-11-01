import numpy as np
import pandas as pd
import os
import shutil

# save all dataframes as csv files in a new folder called citylearn_challenge_2022_phase_1_normalized

# create new folder
if not os.path.exists('data/citylearn_challenge_2022_phase_1_normalized'):
    os.makedirs('data/citylearn_challenge_2022_phase_1_normalized')

# load buildings
buildings = []
for i in range(1, 6):
    buildings.append(pd.read_csv('data/citylearn_challenge_2022_phase_1/building_' + str(i) + '.csv'))

# apply periodic transformation to Month, Hour, Day Type features
for building in buildings:
    building['Month'] = np.sin(2 * np.pi * building['Month'] / 12)
    building['Hour'] = np.sin(2 * np.pi * building['Hour'] / 24)
    building['Day Type'] = np.sin(2 * np.pi * building['Day Type'] / 7)

# min-max normalize all other features over all buildings by concatenating all buildings into one dataframe
# and then min-max normalizing over all buildings, and finally splitting the dataframe back into seperate buildings
n_samples = len(buildings[0])
all_buildings = pd.concat(buildings)
all_buildings = (all_buildings - all_buildings.min()) / (all_buildings.max() - all_buildings.min())
for i in range(5):
    buildings[i] = all_buildings.iloc[i * n_samples:(i + 1) * n_samples]

# save buildings
for i in range(5):
    buildings[i].to_csv('data/citylearn_challenge_2022_phase_1_normalized/building_' + str(i + 1) + '.csv', index=False)


# do same for weather.csv, pricing.csv, and carbon_intensity.csv
weather = pd.read_csv('data/citylearn_challenge_2022_phase_1/weather.csv')
weather = (weather - weather.min()) / (weather.max() - weather.min())
weather.to_csv('data/citylearn_challenge_2022_phase_1_normalized/weather.csv', index=False)

pricing = pd.read_csv('data/citylearn_challenge_2022_phase_1/pricing.csv')
pricing = (pricing - pricing.min()) / (pricing.max() - pricing.min())
pricing.to_csv('data/citylearn_challenge_2022_phase_1_normalized/pricing.csv', index=False)

carbon_intensity = pd.read_csv('data/citylearn_challenge_2022_phase_1/carbon_intensity.csv')
carbon_intensity = (carbon_intensity - carbon_intensity.min()) / (carbon_intensity.max() - carbon_intensity.min())
carbon_intensity.to_csv('data/citylearn_challenge_2022_phase_1_normalized/carbon_intensity.csv', index=False)

# copy schema.json file to new folder
shutil.copy('data/citylearn_challenge_2022_phase_1/schema.json', 'data/citylearn_challenge_2022_phase_1_normalized/schema.json')