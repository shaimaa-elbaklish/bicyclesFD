"""
The Bicycle Fundamental Diagram: Empirical Insights into Bicycle Flow for Sustainable Urban Mobility
-------------------------------------------
Authors:        Shaimaa K. El-Baklish, Ying-Chuan Ni, Kevin Riehl, Anastasios Kouvelas, Michail A. Makridis
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025
Submitted to:   JOURNAL
-------------------------------------------
"""

# #############################################################################
# IMPORTS
# #############################################################################
import os
import gc
import sys
import random
import pathlib
import warnings
warnings.simplefilter('ignore', RuntimeWarning) # Ignore all RuntimeWarnings
warnings.simplefilter('ignore', UserWarning)

import numpy as np
import pandas as pd
import lmfit as lm
import matplotlib.pyplot as plt

from _constants import CRB_Config, SRF_Config

# #############################################################################
# CONSTANTS
# #############################################################################
CRB_Config = CRB_Config()

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

# #############################################################################
# METHODS
# #############################################################################
def compute_pseudo_states_pfd_N2(df, FPS=25.0, NGSIM_flag=False):  
    subdf = df.copy()
    subdf = subdf.sort_values(by=['Vehicle_ID', 'Frame_ID'], ascending=True)
    subdf = subdf.reset_index().drop(columns='index')
    subdf[['Next_Lane_X', 'Next_Space_Hdwy']] = subdf.groupby('Vehicle_ID')[['Lane_X', 'Space_Hdwy']].shift(-1)
    if NGSIM_flag:
        subdf = subdf[(subdf['Preceding'] > 0) & (subdf['Space_Hdwy'] > 0) & ((subdf['Next_Space_Hdwy'] > 0))]
        subdf = subdf[(subdf['v_Class'] == 2) & (subdf['Direction'] == 2) & (subdf['Num_Lanes'] == 1)]
        subdf = subdf[(subdf['Lane_ID'] >= 1)]
    else:
        subdf = subdf.drop(subdf[subdf["Frame_ID"]==subdf["Frame_ID"].max()].index)
    subdf = subdf.dropna()
    
    num_vehicles = 2
    avg_veh_length = 4.5 # m 
    grouped = subdf[["Global_Time", "Vehicle_ID", "Preceding", "Lane_X", "Next_Lane_X", "Space_Hdwy", "Next_Space_Hdwy"]].copy()
    grouped["TTT"] =  (1/FPS) * (num_vehicles-1) / 3600.0 # hour
    grouped["x0"] = grouped["Lane_X"]
    grouped["xt"] = grouped["Next_Lane_X"]
    grouped["xL0"] = grouped["x0"] + grouped["Space_Hdwy"]
    grouped["xLt"] = grouped["xt"] + grouped["Next_Space_Hdwy"]
    grouped["TTD"] = abs(grouped["xt"]-grouped["x0"]) / 1000.0 # km
    grouped["TTD"] = np.round(grouped["TTD"], decimals=6)
    grouped["Area"] = 0.5*(1/FPS/3600.0)*(abs(grouped["xL0"]-grouped["x0"]) + abs(grouped["xLt"]-grouped["xt"]) + 2*avg_veh_length)/1000.0
    grouped["Area"] = grouped["Area"].astype(np.float64).clip(lower=0)
    grouped = grouped[grouped['Area'] > 0]
    grouped["Density"] = grouped["TTT"] / grouped["Area"]
    grouped["Flow"] = grouped["TTD"] / grouped["Area"]
    grouped["Speed"] = grouped["Flow"] / grouped["Density"]
    grouped["Vehicle_IDs"] = grouped[["Preceding", "Vehicle_ID"]].values.tolist()
    
    grouped = grouped[["Global_Time", "Vehicle_IDs", "Density", "Flow", "Speed", "TTT", "TTD"]]
    
    grouped = grouped.dropna()
    return grouped


def aggregate_FD(ts_df, max_density=180.0, bin_width=0.3, min_observations=15):
    ts_df['Density_Bin'] = pd.cut(x=ts_df['Density'], bins=np.arange(0, max_density, bin_width))
    agg_df = ts_df.groupby(["Density_Bin"], observed=False).agg({
        "Density": "mean", 
        "Flow": "mean",
        "Speed": "mean",
        "Density_Bin": "count"
    })
    agg_df = agg_df.rename(
        columns={"Density_Bin": "Num_Observations"}
    )
    agg_df = agg_df.dropna()
    print(agg_df["Num_Observations"].min(), agg_df["Num_Observations"].max())
    print(agg_df["Num_Observations"].mean(), agg_df["Num_Observations"].median())
    agg_df = agg_df[agg_df["Num_Observations"] >= min_observations]
    return agg_df


# #############################################################################
# MAIN: Load CRB dataset
# #############################################################################
pfd_df_all_bikes = pd.read_csv("../data/CRB_PseudoTrafficStates_ALLVideos_V2.txt")
pfd_df_all_bikes = pfd_df_all_bikes.rename(
    columns={'Density': 'Density_Norm', 'Flow': 'Flow_Norm'}
)
pfd_df_all_bikes['Density'] = -1
pfd_df_all_bikes['Flow'] = -1
for video, lane_width in CRB_Config.video_lane_widths.items():
        pfd_df_all_bikes.loc[pfd_df_all_bikes['Video']==video, 'Density'] = pfd_df_all_bikes.loc[pfd_df_all_bikes['Video']==video, 'Density_Norm'] * lane_width
        pfd_df_all_bikes.loc[pfd_df_all_bikes['Video']==video, 'Flow'] = pfd_df_all_bikes.loc[pfd_df_all_bikes['Video']==video, 'Flow_Norm'] * lane_width

# 2.5 m lane width
bike_agg_df_outer = aggregate_FD(pfd_df_all_bikes[pfd_df_all_bikes['Video'].isin(CRB_Config.videos[:-3])], 
                   max_density=400.0, bin_width=0.3, min_observations=50)

# 3.75 m lane width
bike_agg_df_inner = aggregate_FD(pfd_df_all_bikes[pfd_df_all_bikes['Video'].isin(CRB_Config.videos[-3:])], 
                   max_density=400.0, bin_width=0.3, min_observations=50)


v0_bike, d0_bike = 12.0, (1.8 + 1)/1000.0

bike_agg_df_outer['Density_Scaled'] = bike_agg_df_outer['Density'] * d0_bike
bike_agg_df_outer['Flow_Scaled'] = bike_agg_df_outer['Flow'] * d0_bike / v0_bike
bike_agg_df_outer['Speed_Scaled'] = bike_agg_df_outer['Speed'] / v0_bike

bike_agg_df_inner['Density_Scaled'] = bike_agg_df_inner['Density'] * d0_bike
bike_agg_df_inner['Flow_Scaled'] = bike_agg_df_inner['Flow'] * d0_bike / v0_bike
bike_agg_df_inner['Speed_Scaled'] = bike_agg_df_inner['Speed'] / v0_bike

# #############################################################################
# MAIN: Process NGSIM Dataset
# #############################################################################
from _constants import NGSIM_Config

df = pd.read_csv(NGSIM_Config.data_root + f"{NGSIM_Config.locations[0]}/{NGSIM_Config.traj_filenames[0]}")
df['Global_Time'] = df['Global_Time'].astype(float)
df['Global_Time'] = df['Global_Time']/1000.0
df = df.rename(columns={
    'Space_Headway': 'Space_Hdwy', 'Time_Headway': 'Time_Hdwy', 
    'Local_Y': 'Lane_X', 'Local_X': 'Lane_Y'
})
df['Lane_X'] = np.round(df['Lane_X'] / 3.281, decimals=3)  # feet to meter
df['Lane_Y'] = np.round(df['Lane_Y'] / 3.281, decimals=3)  # feet to meter
df['Space_Hdwy'] = np.round(df['Space_Hdwy'] / 3.281, decimals=3)  # feet to meter
df['v_Vel'] = np.round(df['v_Vel'] / 3.281, decimals=3)  # feet/s to m/s
df['Num_Lanes'] = df.groupby('Vehicle_ID')['Lane_ID'].transform('nunique')

ngsim_pfd_df = compute_pseudo_states_pfd_N2(df, FPS=NGSIM_Config.sampling_freq, NGSIM_flag=True)
car_agg_df = aggregate_FD(ngsim_pfd_df, max_density=180.0, bin_width=0.3, min_observations=15)

# #############################################################################
# MAIN: Compare FDs with NGSIM
# #############################################################################
# ngsim_pfd_path = "C:/Users/ShaimaaElBaklish/Desktop/Python_Workspace/PFD_Final/NGSIM_pfd_traffic_states.csv"
# ngsim_pfd_df = pd.read_csv(ngsim_pfd_path)
# ngsim_pfd_df['Density'] = ngsim_pfd_df['Density'] / 1.60934 # veh/mile to veh/km
# ngsim_pfd_df['Speed'] = ngsim_pfd_df['Speed'] * 1.60934 # mph to km/h

# car_agg_df = aggregate_FD(ngsim_pfd_df, max_density=180.0, bin_width=0.3, min_observations=25)

v0_car, d0_car = 40.0,(4.5 + 2)/1000.0

car_agg_df['Density_Scaled'] = car_agg_df['Density'] * d0_car
car_agg_df['Flow_Scaled'] = car_agg_df['Flow'] * d0_car / v0_car
car_agg_df['Speed_Scaled'] = car_agg_df['Speed'] / v0_car


fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].scatter(bike_agg_df_outer['Density_Scaled'], bike_agg_df_outer['Flow_Scaled'], label='Bicycles ($lw$ = 2.5 m)', alpha=0.25)
axs[0].scatter(bike_agg_df_inner['Density_Scaled'], bike_agg_df_inner['Flow_Scaled'], label='Bicycles ($lw$ = 3.75 m)', alpha=0.25)
axs[0].scatter(car_agg_df['Density_Scaled'], car_agg_df['Flow_Scaled'], label='Cars', alpha=0.5)
axs[0].set_xlabel('Scaled Density, $l_0 \\kappa$')
axs[0].set_ylabel('Scaled Flow, $\\frac{l_0}{v_0} q$')
# axs[0].legend()
axs[0].set_ylim([0, 1.5])

axs[1].scatter(bike_agg_df_outer['Density_Scaled'], bike_agg_df_outer['Speed_Scaled'], label='Bicycles ($lw$ = 2.5 m)', alpha=0.25)
axs[1].scatter(bike_agg_df_inner['Density_Scaled'], bike_agg_df_inner['Speed_Scaled'], label='Bicycles ($lw$ = 3.75 m)', alpha=0.25)
axs[1].scatter(car_agg_df['Density_Scaled'], car_agg_df['Speed_Scaled'], label='Cars', alpha=0.5)
axs[1].set_xlabel('Scaled Density, $l_0 \\kappa $')
axs[1].set_ylabel('Scaled Speed, $\\frac{v}{v_0}$')
# axs[1].legend()
axs[1].set_ylim([0, 1.5])

h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, bbox_to_anchor=(0.5, 1.08), loc='upper center', ncol=3, bbox_transform=fig.transFigure)
fig.tight_layout()
fig.savefig('../figures/BFD_CarsComparison_Space_NGSIM.pdf', dpi=300, bbox_inches='tight')

car_agg_df['Density_Eff_Max'] = car_agg_df['Density'] * 4
car_agg_df['Flow_Eff_Max'] = car_agg_df['Flow'] * 4
car_agg_df['Speed_Eff_Max'] = car_agg_df['Speed']

car_agg_df['Density_Eff_Avg'] = car_agg_df['Density'] * 1.5
car_agg_df['Flow_Eff_Avg'] = car_agg_df['Flow'] * 1.5
car_agg_df['Speed_Eff_Avg'] = car_agg_df['Speed']

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].scatter(bike_agg_df_outer['Density'], bike_agg_df_outer['Flow'], label='Bicycles ($lw$ = 2.5 m)', alpha=0.25)
axs[0].scatter(bike_agg_df_inner['Density'], bike_agg_df_inner['Flow'], label='Bicycles ($lw$ = 3.75 m)', alpha=0.25)
axs[0].scatter(car_agg_df['Density_Eff_Max'], car_agg_df['Flow_Eff_Max'], label='Cars ($N_p$ = 4)', alpha=0.5)
axs[0].scatter(car_agg_df['Density_Eff_Avg'], car_agg_df['Flow_Eff_Avg'], label='Cars ($N_p$ = 1.5)', alpha=0.5)
axs[0].set_xlabel('Density, $\\kappa$ [passengers/km]')
axs[0].set_ylabel('Flow, $q$ [passengers/h]')
# axs[0].legend()
axs[0].set_ylim([0, 8000])

axs[1].scatter(bike_agg_df_outer['Density'], bike_agg_df_outer['Speed'], label='Bicycles ($lw$ = 2.5 m)', alpha=0.25)
axs[1].scatter(bike_agg_df_inner['Density'], bike_agg_df_inner['Speed'], label='Bicycles ($lw$ = 3.75 m)', alpha=0.25)
axs[1].scatter(car_agg_df['Density_Eff_Max'], car_agg_df['Speed_Eff_Max'], label='Cars ($N_p$ = 4)', alpha=0.5)
axs[1].scatter(car_agg_df['Density_Eff_Avg'], car_agg_df['Speed_Eff_Avg'], label='Cars ($N_p$ = 1.5)', alpha=0.5)
axs[1].set_xlabel('Density, $\\kappa$ [passengers/km]')
axs[1].set_ylabel('Speed, $v$ [km/h]')
# axs[1].legend()
axs[1].set_ylim([0, 80])

h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, bbox_to_anchor=(0.5, 1.08), loc='upper center', ncol=4, bbox_transform=fig.transFigure)
fig.tight_layout()
fig.savefig('../figures/BFD_CarsComparison_Passenger_NGSIM.pdf', dpi=300, bbox_inches='tight')

