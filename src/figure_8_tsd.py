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
import gc
import sys
import ast
import random
import warnings
warnings.simplefilter('ignore', RuntimeWarning) # Ignore all RuntimeWarnings
warnings.simplefilter('ignore', UserWarning) # Ignore all UserWarnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

from _log_config import create_log_file
create_log_file(logfile = "../logs/CRB_TSE.log")
from _log_config import enable_logging_overwrite
enable_logging_overwrite()

from _constants import CRB_Config
from tools_bfd import estimate_traffic_states

# #############################################################################
# CONSTANTS
# #############################################################################
CRB_Config = CRB_Config()
COMPUTE_PSEUDO_STATES = False

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

video = CRB_Config.videos[-2]
part = CRB_Config.video_parts_X[video][1]

dx, dy, dt = 20*np.pi/180, 0.75, 0.25

MIN_OBS = 5

# #############################################################################
# MAIN
# #############################################################################
pfd_df_all_X = pd.read_csv("../data/CRB_PseudoTrafficStates_ALLVideos_V2.txt") 
pfd_df = pfd_df_all_X[(pfd_df_all_X['Video'] == video) & (pfd_df_all_X['Video_Part'] == part)]
del pfd_df_all_X
gc.collect()

trajectory_df = pd.read_csv(CRB_Config.data_root+ video +"_"+ part +".txt")

# For Polar X
Density_Mat, Flow_Mat, Speed_Mat, Num_Observations_Mat, time_bins, polar_x_bins, polar_y_bins = estimate_traffic_states(
    pfd_df, trajectory_df, dx, dy, dt, CRB_Config.video_lane_widths[video], 
    config=CRB_Config, mode="X"
)

jet = plt.cm.jet
colors = [jet(x) for x in np.linspace(1, 0.5, 256)]
cmap = LinearSegmentedColormap.from_list('GreenToRed', colors, N=256)
space_x_ticks = np.linspace(0, 2*np.pi, 5)
time_ticks = np.linspace(pfd_df['Global_Time'].min(), pfd_df['Global_Time'].max()+dt, 5)

fig, ax = plt.subplots(figsize=(4, 4))
sc = ax.pcolormesh(time_bins, polar_x_bins, Speed_Mat.T, shading='auto', cmap=cmap, vmin=0, vmax=15)
ax.set_yticks(space_x_ticks)
ax.set_yticklabels(np.round(space_x_ticks, 2)) # [f"{y/np.pi:.1f}π" for y in yticks]
ax.set_ylabel('Polar X Position [rad]')
ax.set_xticks(time_ticks)
ax.set_xticklabels(np.round(time_ticks, 1), rotation=0)
ax.set_xlabel('Time [s]')
ax.xaxis.set_ticks_position('bottom')
ax.invert_yaxis()
plt.colorbar(sc, ax=ax, pad=0.04).set_label('Speed [km/h]', rotation=90)
fig.tight_layout()


# For Polar Y
Density_Mat, Flow_Mat, Speed_Mat, Num_Observations_Mat, time_bins, polar_x_bins, polar_y_bins = estimate_traffic_states(
    pfd_df, trajectory_df, dx, dy, dt, CRB_Config.video_lane_widths[video], 
    config=CRB_Config, mode="Y"
)


space_y_ticks = np.linspace(CRB_Config.circle_outer_radius-CRB_Config.video_lane_widths[video], CRB_Config.circle_outer_radius, 6)

fig, ax = plt.subplots(figsize=(4, 4))
sc = ax.pcolormesh(time_bins, polar_y_bins, Speed_Mat.T, shading='auto', cmap=cmap, vmin=0, vmax=15)
ax.set_yticks(space_y_ticks)
ax.set_yticklabels(np.round(space_y_ticks, 2))
ax.set_ylabel('Polar X Position [rad]')
ax.set_xticks(time_ticks)
ax.set_xticklabels(np.round(time_ticks, 1), rotation=0)
ax.set_xlabel('Time [s]')
ax.xaxis.set_ticks_position('bottom')
ax.set_ylim([CRB_Config.circle_outer_radius-CRB_Config.video_lane_widths[video], CRB_Config.circle_outer_radius])
plt.colorbar(sc, ax=ax, pad=0.04).set_label('Speed [km/h]', rotation=90)
fig.tight_layout()

sys.exit(1)


# pfd_df['Time_Bin'] = pd.cut(pfd_df['Global_Time'], 
#                             bins=np.arange(pfd_df['Global_Time'].min(), pfd_df['Global_Time'].max()+dt, dt))
# pfd_df['Polar_X_Bin'] = pd.cut(pfd_df['Polar_X'], 
#                                bins=np.arange(0, 2*np.pi+dx, dx))
# pfd_df['Polar_Y_Bin'] = pd.cut(pfd_df['Polar_Y'], 
#                                bins=np.arange(CRB_Config.circle_outer_radius-CRB_Config.video_lane_widths[video], 
#                                               CRB_Config.circle_outer_radius+dy, dy))

# # For Polar X
# grouped = pfd_df.groupby(by=['Time_Bin', 'Polar_X_Bin'], observed=False).agg(
#     Num_Observations=pd.NamedAgg(column="Density", aggfunc="count"),
#     Density=pd.NamedAgg(column="Density", aggfunc="mean"),
#     Flow=pd.NamedAgg(column="Flow", aggfunc="mean"),
#     Speed=pd.NamedAgg(column="Speed", aggfunc="mean"),
# )
# grouped = grouped.reset_index().dropna()

# n_t_rows = int((pfd_df['Global_Time'].max()-pfd_df['Global_Time'].min())/dt) + 1
# n_x_cols = int(2*np.pi/dx) + 1
# Density_Mat = np.empty(shape=(n_t_rows, n_x_cols))
# Flow_Mat = np.empty(shape=(n_t_rows, n_x_cols))
# Speed_Mat = np.empty(shape=(n_t_rows, n_x_cols))
# Num_Observations_Mat = np.zeros(shape=(n_t_rows, n_x_cols))
# Density_Mat.fill(np.nan)
# Flow_Mat.fill(np.nan)
# # Speed_Mat.fill(np.nan)
# Speed_Mat.fill(15.0)
# for _, row in grouped.iterrows():
#     it = int((row['Time_Bin'].left - pfd_df['Global_Time'].min())/dt)
#     jx = int(row['Polar_X_Bin'].left/dx)
#     if it >= n_t_rows or jx >= n_x_cols:
#         continue
#     if row['Num_Observations'] < MIN_OBS:
#         continue
#     Density_Mat[it, jx] = row['Density']
#     Flow_Mat[it, jx] = row['Flow']
#     Speed_Mat[it, jx] = row['Speed']
#     Num_Observations_Mat[it, jx] = row['Num_Observations']
# Density_Mat = np.flip(Density_Mat.T, axis=0)
# Flow_Mat = np.flip(Flow_Mat.T, axis=0)
# Speed_Mat = np.flip(Speed_Mat.T, axis=0)
# Num_Observations_Mat = np.flip(Num_Observations_Mat.T, axis=0)

# jet = plt.cm.jet
# colors = [jet(x) for x in np.linspace(1, 0.5, 256)]
# cmap = LinearSegmentedColormap.from_list('GreenToRed', colors, N=256)
# space_x_ticks = np.linspace(0, 2*np.pi, 5)
# space_x_ticks = space_x_ticks[::-1]
# time_ticks = np.linspace(0, (pfd_df['Global_Time'].max()-pfd_df['Global_Time'].min())+dt, 5)

# fig, ax = plt.subplots(figsize=(4, 4))
# sc = ax.matshow(Speed_Mat, cmap=cmap, vmin=0, vmax=15, aspect='auto')
# ax.set_yticks(np.linspace(0, n_x_cols, len(space_x_ticks)))
# ax.set_yticklabels(np.round(space_x_ticks, 2))
# ax.set_ylabel('Polar X Position [rad]')
# ax.set_xticks(np.linspace(0, n_t_rows, len(time_ticks)))
# ax.set_xticklabels(np.round(time_ticks, 1), rotation=0)
# ax.set_xlabel('Time [s]')
# ax.xaxis.set_ticks_position('bottom')
# plt.colorbar(sc, ax=ax, pad=0.04).set_label('Speed [km/h]', rotation=90)
# fig.tight_layout()


# # For Polar_Y
# grouped = pfd_df.groupby(by=['Time_Bin', 'Polar_Y_Bin'], observed=False).agg(
#     Num_Observations=pd.NamedAgg(column="Density", aggfunc="count"),
#     Density=pd.NamedAgg(column="Density", aggfunc="mean"),
#     Flow=pd.NamedAgg(column="Flow", aggfunc="mean"),
#     Speed=pd.NamedAgg(column="Speed", aggfunc="mean"),
# )
# grouped = grouped.reset_index().dropna()

# n_t_rows = int((pfd_df['Global_Time'].max()-pfd_df['Global_Time'].min())/dt) + 1
# n_y_cols = int(CRB_Config.video_lane_widths[video]/dy) + 1
# Density_Mat = np.empty(shape=(n_t_rows, n_y_cols))
# Flow_Mat = np.empty(shape=(n_t_rows, n_y_cols))
# Speed_Mat = np.empty(shape=(n_t_rows, n_y_cols))
# Num_Observations_Mat = np.zeros(shape=(n_t_rows, n_y_cols))
# Density_Mat.fill(np.nan)
# Flow_Mat.fill(np.nan)
# # Speed_Mat.fill(np.nan)
# Speed_Mat.fill(15.0)
# for _, row in grouped.iterrows():
#     it = int((row['Time_Bin'].left - pfd_df['Global_Time'].min())/dt)
#     jx = int((row['Polar_Y_Bin'].left - CRB_Config.circle_outer_radius + CRB_Config.video_lane_widths[video])/dy)
#     if it >= n_t_rows or jx >= n_y_cols:
#         continue
#     if row['Num_Observations'] < MIN_OBS:
#         continue
#     Density_Mat[it, jx] = row['Density']
#     Flow_Mat[it, jx] = row['Flow']
#     Speed_Mat[it, jx] = row['Speed']
#     Num_Observations_Mat[it, jx] = row['Num_Observations']
# Density_Mat = np.flip(Density_Mat.T, axis=0)
# Flow_Mat = np.flip(Flow_Mat.T, axis=0)
# Speed_Mat = np.flip(Speed_Mat.T, axis=0)
# Num_Observations_Mat = np.flip(Num_Observations_Mat.T, axis=0)

# jet = plt.cm.jet
# colors = [jet(x) for x in np.linspace(1, 0.5, 256)]
# cmap = LinearSegmentedColormap.from_list('GreenToRed', colors, N=256)
# space_y_ticks = np.linspace(CRB_Config.circle_outer_radius-CRB_Config.video_lane_widths[video], CRB_Config.circle_outer_radius+dy, 5)
# space_y_ticks = space_y_ticks[::-1]
# time_ticks = np.linspace(0, (pfd_df['Global_Time'].max()-pfd_df['Global_Time'].min())+dt, 5)

# fig, ax = plt.subplots(figsize=(4, 4))
# sc = ax.matshow(Speed_Mat, cmap=cmap, vmin=0, vmax=15, aspect='auto')
# ax.set_yticks(np.linspace(0, n_y_cols, len(space_y_ticks)))
# ax.set_yticklabels(np.round(space_y_ticks, 2))
# ax.set_ylabel('Polar Y Position [m]')
# ax.set_xticks(np.linspace(0, n_t_rows, len(time_ticks)))
# ax.set_xticklabels(np.round(time_ticks, 1), rotation=0)
# ax.set_xlabel('Time [s]')
# ax.xaxis.set_ticks_position('bottom')
# plt.colorbar(sc, ax=ax, pad=0.04).set_label('Speed [km/h]', rotation=90)
# fig.tight_layout()

# sys.exit(1)

# # For Polar_X and Polar_Y
# grouped = pfd_df.groupby(by=['Time_Bin', 'Polar_X_Bin', 'Polar_Y_Bin'], observed=False).agg(
#     Num_Observations=pd.NamedAgg(column="Density", aggfunc="count"),
#     Density=pd.NamedAgg(column="Density", aggfunc="mean"),
#     Flow=pd.NamedAgg(column="Flow", aggfunc="mean"),
#     Speed=pd.NamedAgg(column="Speed", aggfunc="mean"),
# )
# grouped = grouped.reset_index().dropna()

# Density_Mat = np.empty(shape=(n_t_rows, n_x_cols, n_y_cols))
# Flow_Mat = np.empty(shape=(n_t_rows, n_x_cols, n_y_cols))
# Speed_Mat = np.empty(shape=(n_t_rows, n_x_cols, n_y_cols))
# Num_Observations_Mat = np.zeros(shape=(n_t_rows, n_x_cols, n_y_cols))
# Density_Mat.fill(np.nan)
# Flow_Mat.fill(np.nan)
# Speed_Mat.fill(np.nan)
# for _, row in grouped.iterrows():
#     it = int((row['Time_Bin'].left - pfd_df['Global_Time'].min())/dt)
#     jx = int(row['Polar_X_Bin'].left/dx)
#     jy = int((row['Polar_Y_Bin'].left - CRB_Config.circle_outer_radius + CRB_Config.video_lane_widths[video])/dy)
#     if it >= n_t_rows or jx >= n_x_cols or jy >= n_y_cols:
#         continue
#     if row['Num_Observations'] < MIN_OBS:
#         continue
#     Density_Mat[it, jx, jy] = row['Density']
#     Flow_Mat[it, jx, jy] = row['Flow']
#     Speed_Mat[it, jx, jy] = row['Speed']
#     Num_Observations_Mat[it, jx, jy] = row['Num_Observations']
# Density_Mat = np.flip(Density_Mat.T, axis=0)
# Flow_Mat = np.flip(Flow_Mat.T, axis=0)
# Speed_Mat = np.flip(Speed_Mat.T, axis=0)
# Num_Observations_Mat = np.flip(Num_Observations_Mat.T, axis=0)

