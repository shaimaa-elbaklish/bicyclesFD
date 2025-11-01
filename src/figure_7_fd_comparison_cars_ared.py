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
import sys
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lmfit as lm
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection

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
def compute_pseudo_states_pfd_N2(df, FPS=25.0, ARED_flag=False):  
    subdf = df.copy()
    subdf = subdf.sort_values(by=['Vehicle_ID', 'Frame_ID'], ascending=True)
    subdf = subdf.reset_index().drop(columns='index')
    subdf[['Next_Lane_X', 'Next_Space_Hdwy']] = subdf.groupby('Vehicle_ID')[['Lane_X', 'Space_Hdwy']].shift(-1)
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


def plot_resetted_tsd(df):
    plt.rc('font', family='sans-serif') 
    plt.rc('font', serif='Arial') 
    fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
    cmap = plt.get_cmap('rainbow_r') # Choose colormap (e.g., plasma, magma, inferno)
    norm = plt.Normalize(vmin=df["cam_velocity"].min(), vmax=df["cam_velocity"].max())  # Global color scaling
    unique_vehicles = df['veh_id'].unique()
    for vehicle in unique_vehicles:
        df_sub = df[df["veh_id"]==vehicle]
        global_time = df_sub["timestamp"].tolist()
        lane_x = df_sub["position"].tolist()
        velocity = df_sub["cam_velocity"].tolist()
        # Split to parts
        plots = []
        current_x = []
        current_time = []
        current_vel = []
        ctr = 0
        for t, x, vel in zip(global_time, lane_x, velocity):
            if x - ctr*ARED_Config.track_circumference > ARED_Config.track_circumference:
                if len(current_x)>0:
                    plots.append([current_time, current_x, current_vel])
                current_x = []
                current_time = []
                current_vel = []
                ctr += 1
            current_x.append(x - ctr*ARED_Config.track_circumference)
            current_vel.append(vel)
            current_time.append(t)
        # Last segment
        if current_x:
            plots.append([current_time, current_x, current_vel])
        # Plot
        for part in plots:
            x = np.array(part[0])
            y = np.array(part[1]) 
            velocities = np.array(part[2]) 
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, array=velocities)
            line = ax.add_collection(lc)

    ax.autoscale_view()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Vehicle Position [m]\n(Resetted every circumference)')

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        label='Velocity [m/s]'
    )

    fig.tight_layout()
    plt.show()

# #############################################################################
# MAIN: Process ARED Dataset
# #############################################################################
from _constants import ARED_Config

ared_pfd_df_all = None
for i in range(len(ARED_Config.traj_filenames)):
    df = pd.read_csv(f"{ARED_Config.data_root}/{ARED_Config.traj_filenames[i]}")
    df_mod = None
    unique_vehicles = df['veh_id'].unique()
    veh_length = {}
    for veh_id in unique_vehicles:
        veh_df = df[df['veh_id'] == veh_id].copy()
        veh_df = veh_df.sort_values(by='timestamp', ascending=True)
        veh_df['distance_travelled'] = np.cumsum(veh_df['cam_velocity'] / ARED_Config.sampling_freq)
        veh_length[veh_id] = veh_df['length'].iloc[0]
        if df_mod is None:
            df_mod = veh_df.copy()
        else:
            df_mod = pd.concat((df_mod, veh_df), ignore_index=True)
    df = df_mod
    
    first_veh_idx = df.loc[df['timestamp'] == df['timestamp'].min(), 'cam_distance'].idxmin()
    veh_id = df.loc[first_veh_idx, 'veh_id']
    init_offset = {}
    init_offset[veh_id] = 0
    prev_offset = 0
    unique_vehicles = set(unique_vehicles)
    while len(unique_vehicles) >= 1:
        veh_df = df[(df['timestamp'] == df['timestamp'].min()) & (df['veh_id'] == veh_id)].copy()
        prec_veh_id = veh_df['leader_id'].iloc[0]
        if prec_veh_id == df.loc[first_veh_idx, 'veh_id']:
            break
        init_offset[prec_veh_id] = prev_offset + veh_df['cam_leader_gap'].iloc[0] + veh_length[prec_veh_id]
        prev_offset = init_offset[prec_veh_id]
        veh_id = prec_veh_id
        unique_vehicles = unique_vehicles - set([veh_id])
    df['offset'] = df['veh_id'].map(init_offset)
    df['position'] = df['distance_travelled'] + df['offset']
    df['position'] = df['position'].round(decimals=4)
    # df['position'] = df['position'].round(decimals=4).mod(ARED_Config.track_circumference)
    
    df['Frame_ID'] = df['timestamp'] * ARED_Config.sampling_freq
    df['Frame_ID'] = df['Frame_ID'].round(decimals=0).astype(int)
    df['leader_length'] = df['leader_id'].map(veh_length)
    df['Space_Hdwy'] = df['cam_leader_gap'] + df['leader_length']
    df.loc[df['cam_leader_gap'] < 0, 'Space_Hdwy'] += ARED_Config.track_circumference
    df = df.rename(columns={
        'veh_id': 'Vehicle_ID', 'position': 'Lane_X', 'timestamp': 'Global_Time', 'leader_id': 'Preceding'
    })
    ared_pfd_df = compute_pseudo_states_pfd_N2(df, FPS=ARED_Config.sampling_freq)
    ared_pfd_df['exp_id'] = df['exp_id'].iloc[0]
    if ared_pfd_df_all is None:
        ared_pfd_df_all = ared_pfd_df.copy()
    else:
        ared_pfd_df_all = pd.concat((ared_pfd_df_all, ared_pfd_df), ignore_index=True)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

exp_set = [s.split('.')[0] for s in ARED_Config.traj_filenames[:5]]
car_agg_df = aggregate_FD(ared_pfd_df_all[ared_pfd_df_all['exp_id'].isin(exp_set)], 
                          max_density=180.0, bin_width=0.3, min_observations=50)
axs[0].scatter(car_agg_df['Density'], car_agg_df['Flow'], label='Instruction I', alpha=0.75)
axs[1].scatter(car_agg_df['Density'], car_agg_df['Speed'], label='Instruction I', alpha=0.75)

exp_set = [s.split('.')[0] for s in ARED_Config.traj_filenames[-3:]]
car_agg_df = aggregate_FD(ared_pfd_df_all[ared_pfd_df_all['exp_id'].isin(exp_set)], 
                          max_density=180.0, bin_width=0.3, min_observations=50)
axs[0].scatter(car_agg_df['Density'], car_agg_df['Flow'], label='Instruction II', alpha=0.5)
axs[1].scatter(car_agg_df['Density'], car_agg_df['Speed'], label='Instruction II', alpha=0.5)

car_agg_df = aggregate_FD(ared_pfd_df_all, max_density=180.0, bin_width=0.3, min_observations=50)
axs[0].scatter(car_agg_df['Density'], car_agg_df['Flow'], label='ALL', alpha=0.25)
axs[1].scatter(car_agg_df['Density'], car_agg_df['Speed'], label='ALL', alpha=0.25)

axs[0].set_xlabel('Density [veh/km]')
axs[0].set_ylabel('Flow [veh/h]')
axs[0].legend()
axs[0].set_ylim([0, 2000])
axs[1].set_xlabel('Density [veh/km]')
axs[1].set_ylabel('Speed [km/h]')
axs[1].set_ylim([0, 30])
fig.tight_layout()
# sys.exit(1)

exp_set = [s.split('.')[0] for s in ARED_Config.traj_filenames[-3:]]
car_agg_df = aggregate_FD(ared_pfd_df_all[ared_pfd_df_all['exp_id'].isin(exp_set)], 
                          max_density=180.0, bin_width=0.3, min_observations=50)

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

# #############################################################################
# MAIN: Make Transformations
# #############################################################################
# PASSENGER EFFICIENCY
car_agg_df['Density_Eff_Max'] = car_agg_df['Density'] * 4
car_agg_df['Flow_Eff_Max'] = car_agg_df['Flow'] * 4
car_agg_df['Speed_Eff_Max'] = car_agg_df['Speed']

car_agg_df['Density_Eff_Avg'] = car_agg_df['Density'] * 1.5
car_agg_df['Flow_Eff_Avg'] = car_agg_df['Flow'] * 1.5
car_agg_df['Speed_Eff_Avg'] = car_agg_df['Speed']

car_agg_df['Density_Eff_Opt'] = car_agg_df['Density'] * 2.5
car_agg_df['Flow_Eff_Opt'] = car_agg_df['Flow'] * 2.5
car_agg_df['Speed_Eff_Opt'] = car_agg_df['Speed']

# SPACE UTILITY
v0_car, d0_car = 25.0, (4.5 + 2)/1000.0
v0_bike, d0_bike = 12.0, (1.8 + 1)/1000.0

car_agg_df['Density_Scaled'] = car_agg_df['Density'] * d0_car
car_agg_df['Flow_Scaled'] = car_agg_df['Flow'] * d0_car / v0_car
car_agg_df['Speed_Scaled'] = car_agg_df['Speed'] / v0_car

bike_agg_df_outer['Density_Scaled'] = bike_agg_df_outer['Density'] * d0_bike
bike_agg_df_outer['Flow_Scaled'] = bike_agg_df_outer['Flow'] * d0_bike / v0_bike
bike_agg_df_outer['Speed_Scaled'] = bike_agg_df_outer['Speed'] / v0_bike

bike_agg_df_inner['Density_Scaled'] = bike_agg_df_inner['Density'] * d0_bike
bike_agg_df_inner['Flow_Scaled'] = bike_agg_df_inner['Flow'] * d0_bike / v0_bike
bike_agg_df_inner['Speed_Scaled'] = bike_agg_df_inner['Speed'] / v0_bike

# #############################################################################
# MAIN: Make Big Figure
# #############################################################################
fig, axs = plt.subplots(2, 3, figsize=(3*4, 2*4))

fs_labels, fs_ticks, fs_title = 12, 12, 14
b_col_2p5, b_col_3p75 = 'tab:blue', 'tab:orange'
c_col_1, c_col_1p5, c_col_2p5, c_col_4 = 'tab:green', 'tab:red', 'tab:purple', 'tab:brown'


# ORIGINAL
axs[0, 0].scatter(bike_agg_df_outer['Density'], bike_agg_df_outer['Flow'], label='Bicycles ($lw$ = 2.5 m)', alpha=0.25, color=b_col_2p5)
axs[0, 0].scatter(bike_agg_df_inner['Density'], bike_agg_df_inner['Flow'], label='Bicycles ($lw$ = 3.75 m)', alpha=0.25, color=b_col_3p75)
axs[0, 0].scatter(car_agg_df['Density'], car_agg_df['Flow'], label='Cars ($N_p$ = 1)', alpha=0.5, color=c_col_1)
axs[0, 0].set_xlabel('Density [km$^{-1}$]', fontsize=fs_labels)
axs[0, 0].set_ylabel('Flow [h$^{-1}$]', fontsize=fs_labels)
axs[0, 0].set_ylim([0, 3500])
axs[0, 0].set_xlim([0, 400])
axs[0, 0].set_yticks(np.linspace(0, 3000, 4))
axs[0, 0].set_title('(a) Original Data', fontweight='bold', fontsize=fs_title)
axs[0, 0].tick_params(axis='both', labelsize=fs_ticks)

axs[1, 0].scatter(bike_agg_df_outer['Density'], bike_agg_df_outer['Speed'], label='Bicycles ($lw$ = 2.5 m)', alpha=0.25, color=b_col_2p5)
axs[1, 0].scatter(bike_agg_df_inner['Density'], bike_agg_df_inner['Speed'], label='Bicycles ($lw$ = 3.75 m)', alpha=0.25, color=b_col_3p75)
axs[1, 0].scatter(car_agg_df['Density'], car_agg_df['Speed'], label='Cars ($N_p$ = 1)', alpha=0.5, color=c_col_1)
axs[1, 0].set_xlabel('Density [km$^{-1}$]', fontsize=fs_labels)
axs[1, 0].set_ylabel('Speed [km/h]', fontsize=fs_labels)
axs[1, 0].set_ylim([0, 30])
axs[1, 0].set_xlim([0, 400])
axs[1, 0].tick_params(axis='both', labelsize=fs_ticks)


# PASSENGER EFFICIENCY
axs[0, 1].scatter(bike_agg_df_outer['Density'], bike_agg_df_outer['Flow'], label='Bicycles ($lw$ = 2.5 m)', alpha=0.25, color=b_col_2p5)
axs[0, 1].scatter(bike_agg_df_inner['Density'], bike_agg_df_inner['Flow'], label='Bicycles ($lw$ = 3.75 m)', alpha=0.25, color=b_col_3p75)
axs[0, 1].scatter(car_agg_df['Density_Eff_Max'], car_agg_df['Flow_Eff_Max'], label='Cars ($N_p$ = 4)', alpha=0.5, color=c_col_4)
axs[0, 1].scatter(car_agg_df['Density_Eff_Opt'], car_agg_df['Flow_Eff_Opt'], label='Cars ($N_p$ = 2.5)', alpha=0.5, color=c_col_2p5)
axs[0, 1].scatter(car_agg_df['Density_Eff_Avg'], car_agg_df['Flow_Eff_Avg'], label='Cars ($N_p$ = 1.5)', alpha=0.5, color=c_col_1p5)
axs[0, 1].set_xlabel('Density, $\\kappa$ [passengers/km]', fontsize=fs_labels)
axs[0, 1].set_ylabel('Flow, $q$ [passengers/h]', fontsize=fs_labels)
axs[0, 1].set_ylim([0, 5500])
axs[0, 1].set_yticks(np.linspace(0, 5000, 6))
axs[0, 1].set_title('(b) Passenger Efficiency Perspective', fontweight='bold', fontsize=fs_title)
axs[0, 1].tick_params(axis='both', labelsize=fs_ticks)

axs[1, 1].scatter(bike_agg_df_outer['Density'], bike_agg_df_outer['Speed'], label='Bicycles ($lw$ = 2.5 m)', alpha=0.25, color=b_col_2p5)
axs[1, 1].scatter(bike_agg_df_inner['Density'], bike_agg_df_inner['Speed'], label='Bicycles ($lw$ = 3.75 m)', alpha=0.25, color=b_col_3p75)
axs[1, 1].scatter(car_agg_df['Density_Eff_Max'], car_agg_df['Speed_Eff_Max'], label='Cars ($N_p$ = 4)', alpha=0.5, color=c_col_4)
axs[1, 1].scatter(car_agg_df['Density_Eff_Opt'], car_agg_df['Speed_Eff_Opt'], label='Cars ($N_p$ = 2.5)', alpha=0.5, color=c_col_2p5)
axs[1, 1].scatter(car_agg_df['Density_Eff_Avg'], car_agg_df['Speed_Eff_Avg'], label='Cars ($N_p$ = 1.5)', alpha=0.5, color=c_col_1p5)
axs[1, 1].set_xlabel('Density, $\\kappa$ [passengers/km]', fontsize=fs_labels)
axs[1, 1].set_ylabel('Speed, $v$ [km/h]', fontsize=fs_labels)
axs[1, 1].set_ylim([0, 30])
axs[1, 1].tick_params(axis='both', labelsize=fs_ticks)


# SPACE UTILITY
axs[0, 2].scatter(bike_agg_df_outer['Density_Scaled'], bike_agg_df_outer['Flow_Scaled'], label='Bicycles ($lw$ = 2.5 m)', alpha=0.25, color=b_col_2p5)
axs[0, 2].scatter(bike_agg_df_inner['Density_Scaled'], bike_agg_df_inner['Flow_Scaled'], label='Bicycles ($lw$ = 3.75 m)', alpha=0.25, color=b_col_3p75)
axs[0, 2].scatter(car_agg_df['Density_Scaled'], car_agg_df['Flow_Scaled'], label='Cars ($N_p$ = 1)', alpha=0.5, color=c_col_1)
axs[0, 2].set_xlabel('Scaled Density, $l_0 \\kappa$', fontsize=fs_labels)
axs[0, 2].set_ylabel('Scaled Flow, $\\frac{l_0}{v_0} q$', fontsize=fs_labels)
axs[0, 2].set_ylim([0, 1.0])
axs[0, 2].set_xticks(np.arange(0, 1.01, 0.2))
axs[0, 2].set_title('(c) Space Utility Perspective', fontweight='bold', fontsize=fs_title)
axs[0, 2].tick_params(axis='both', labelsize=fs_ticks)

axs[1, 2].scatter(bike_agg_df_outer['Density_Scaled'], bike_agg_df_outer['Speed_Scaled'], label='Bicycles ($lw$ = 2.5 m)', alpha=0.25, color=b_col_2p5)
axs[1, 2].scatter(bike_agg_df_inner['Density_Scaled'], bike_agg_df_inner['Speed_Scaled'], label='Bicycles ($lw$ = 3.75 m)', alpha=0.25, color=b_col_3p75)
axs[1, 2].scatter(car_agg_df['Density_Scaled'], car_agg_df['Speed_Scaled'], label='Cars ($N_p$ = 1)', alpha=0.5, color=c_col_1)
axs[1, 2].set_xlabel('Scaled Density, $l_0 \\kappa $', fontsize=fs_labels)
axs[1, 2].set_ylabel('Scaled Speed, $\\frac{v}{v_0}$', fontsize=fs_labels)
axs[1, 2].set_ylim([0, 1.3])
axs[1, 2].set_xticks(np.arange(0, 1.01, 0.2))
axs[1, 2].tick_params(axis='both', labelsize=fs_ticks)


h, l = axs[0, 1].get_legend_handles_labels()
h2, l2 = axs[0, 0].get_legend_handles_labels()
h, l = h + [h2[-1]], l + [l2[-1]]
fig.legend(h, l, bbox_to_anchor=(0.5, -0.05), loc='lower center', ncol=6, bbox_transform=fig.transFigure, fontsize=fs_ticks)
fig.tight_layout()
fig.savefig('../figures/BFD_CarsComparison_ARED.pdf', dpi=300, bbox_inches='tight')
sys.exit(1)


# #############################################################################
# MAIN: Compare FDs: Original
# #############################################################################
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].scatter(bike_agg_df_outer['Density'], bike_agg_df_outer['Flow'], label='Bicycles ($lw$ = 2.5 m)', alpha=0.25)
axs[0].scatter(bike_agg_df_inner['Density'], bike_agg_df_inner['Flow'], label='Bicycles ($lw$ = 3.75 m)', alpha=0.25)
axs[0].scatter(car_agg_df['Density'], car_agg_df['Flow'], label='Cars', alpha=0.5)
axs[0].set_xlabel('Density [km$^{-1}$]')
axs[0].set_ylabel('Flow [h$^{-1}$]')
# axs[0].legend()
axs[0].set_ylim([0, 4000])

axs[1].scatter(bike_agg_df_outer['Density'], bike_agg_df_outer['Speed'], label='Bicycles ($lw$ = 2.5 m)', alpha=0.25)
axs[1].scatter(bike_agg_df_inner['Density'], bike_agg_df_inner['Speed'], label='Bicycles ($lw$ = 3.75 m)', alpha=0.25)
axs[1].scatter(car_agg_df['Density'], car_agg_df['Speed'], label='Cars', alpha=0.5)
axs[1].set_xlabel('Density [km$^{-1}$]')
axs[1].set_ylabel('Speed [km/h]')
# axs[1].legend()
axs[1].set_ylim([0, 30])

h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, bbox_to_anchor=(0.5, 1.08), loc='upper center', ncol=3, bbox_transform=fig.transFigure)
fig.tight_layout()

# #############################################################################
# MAIN: Compare FDs: Space Utility
# #############################################################################
v0_car, d0_car = 25.0, (4.5 + 2)/1000.0

car_agg_df['Density_Scaled'] = car_agg_df['Density'] * d0_car
car_agg_df['Flow_Scaled'] = car_agg_df['Flow'] * d0_car / v0_car
car_agg_df['Speed_Scaled'] = car_agg_df['Speed'] / v0_car

v0_bike, d0_bike = 12.0, (1.8 + 1)/1000.0

bike_agg_df_outer['Density_Scaled'] = bike_agg_df_outer['Density'] * d0_bike
bike_agg_df_outer['Flow_Scaled'] = bike_agg_df_outer['Flow'] * d0_bike / v0_bike
bike_agg_df_outer['Speed_Scaled'] = bike_agg_df_outer['Speed'] / v0_bike

bike_agg_df_inner['Density_Scaled'] = bike_agg_df_inner['Density'] * d0_bike
bike_agg_df_inner['Flow_Scaled'] = bike_agg_df_inner['Flow'] * d0_bike / v0_bike
bike_agg_df_inner['Speed_Scaled'] = bike_agg_df_inner['Speed'] / v0_bike


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
fig.savefig('../figures/BFD_CarsComparison_Space_ARED.pdf', dpi=300, bbox_inches='tight')

# #############################################################################
# MAIN: Compare FDs: Passenger Perspective
# #############################################################################
car_agg_df['Density_Eff_Max'] = car_agg_df['Density'] * 4
car_agg_df['Flow_Eff_Max'] = car_agg_df['Flow'] * 4
car_agg_df['Speed_Eff_Max'] = car_agg_df['Speed']

car_agg_df['Density_Eff_Avg'] = car_agg_df['Density'] * 1.5
car_agg_df['Flow_Eff_Avg'] = car_agg_df['Flow'] * 1.5
car_agg_df['Speed_Eff_Avg'] = car_agg_df['Speed']

car_agg_df['Density_Eff_Opt'] = car_agg_df['Density'] * 2.5
car_agg_df['Flow_Eff_Opt'] = car_agg_df['Flow'] * 2.5
car_agg_df['Speed_Eff_Opt'] = car_agg_df['Speed']

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].scatter(bike_agg_df_outer['Density'], bike_agg_df_outer['Flow'], label='Bicycles ($lw$ = 2.5 m)', alpha=0.25)
axs[0].scatter(bike_agg_df_inner['Density'], bike_agg_df_inner['Flow'], label='Bicycles ($lw$ = 3.75 m)', alpha=0.25)
axs[0].scatter(car_agg_df['Density_Eff_Max'], car_agg_df['Flow_Eff_Max'], label='Cars ($N_p$ = 4)', alpha=0.5)
axs[0].scatter(car_agg_df['Density_Eff_Opt'], car_agg_df['Flow_Eff_Opt'], label='Cars ($N_p$ = 2.5)', alpha=0.5)
axs[0].scatter(car_agg_df['Density_Eff_Avg'], car_agg_df['Flow_Eff_Avg'], label='Cars ($N_p$ = 1.5)', alpha=0.5)
axs[0].set_xlabel('Density, $\\kappa$ [passengers/km]')
axs[0].set_ylabel('Flow, $q$ [passengers/h]')
# axs[0].legend()
axs[0].set_ylim([0, 6000])

axs[1].scatter(bike_agg_df_outer['Density'], bike_agg_df_outer['Speed'], label='Bicycles ($lw$ = 2.5 m)', alpha=0.25)
axs[1].scatter(bike_agg_df_inner['Density'], bike_agg_df_inner['Speed'], label='Bicycles ($lw$ = 3.75 m)', alpha=0.25)
axs[1].scatter(car_agg_df['Density_Eff_Max'], car_agg_df['Speed_Eff_Max'], label='Cars ($N_p$ = 4)', alpha=0.5)
axs[1].scatter(car_agg_df['Density_Eff_Opt'], car_agg_df['Speed_Eff_Opt'], label='Cars ($N_p$ = 2.5)', alpha=0.5)
axs[1].scatter(car_agg_df['Density_Eff_Avg'], car_agg_df['Speed_Eff_Avg'], label='Cars ($N_p$ = 1.5)', alpha=0.5)
axs[1].set_xlabel('Density, $\\kappa$ [passengers/km]')
axs[1].set_ylabel('Speed, $v$ [km/h]')
# axs[1].legend()
axs[1].set_ylim([0, 35])

h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=3, bbox_transform=fig.transFigure)
fig.tight_layout()
fig.savefig('../figures/BFD_CarsComparison_Passenger_ARED.pdf', dpi=300, bbox_inches='tight')