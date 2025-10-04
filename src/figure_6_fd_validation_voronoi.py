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
import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon
from pedpy import PEDPY_BLUE, PEDPY_ORANGE, DENSITY_COL
from pedpy import load_trajectory, plot_trajectories, plot_measurement_setup
from pedpy import WalkableArea, MeasurementArea, Cutoff
from pedpy import compute_individual_voronoi_polygons, compute_voronoi_density
from pedpy import compute_individual_speed, compute_voronoi_speed
from pedpy import SpeedCalculation, plot_speed, plot_density, plot_voronoi_cells

from _log_config import create_log_file
create_log_file(logfile = "../logs/CRB_FD_Validation.log")
from _log_config import logger, enable_logging_overwrite, enable_logging_append
enable_logging_overwrite()

from _constants import CRB_Config
from tools_bfd import aggregate_fd

# #############################################################################
# CONSTANTS
# #############################################################################
CRB_Config = CRB_Config()
COMPUTE_PSEUDO_STATES = False

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

# for Voronoi method
NUM_SECTORS = 5

# for SSD method
MIN_BIKES = 2

# for aggregation
MIN_OBS = 50

# #############################################################################
# FUNCTIONS
# #############################################################################
def prepare_data_pedpy(df, lane_width, config):
    df = df[df["Polar_Y"] >= config.circle_outer_radius - lane_width]
    df = df[df["Polar_Y"] <= config.circle_outer_radius]
    df = df[["Vehicle_ID", "Frame_ID", "Cartesian_X", "Cartesian_Y"]]
    df = df.rename(columns={
        "Frame_ID": "frame",
        "Cartesian_X": "X",
        "Cartesian_Y": "Y"
    })
    df["ID"] = df["Vehicle_ID"].str.split("_")
    df["ID"] = df["ID"].apply(lambda x: x[-1])
    df = df.astype({"ID": "int"})
    df =  df.drop(columns=["Vehicle_ID"])
    df["Z"] = 1.1 # average bicycle height
    df = df[["ID", "frame", "X", "Y", "Z"]]
    return df


def filter_traj_inside(traj, walkable_area_poly):
    mask = traj.apply(lambda row: walkable_area_poly.contains(Point(row["x"], row["y"])), axis=1)
    return traj[mask]


def define_measurement_setup(num_sectors, lane_width):
    center_point = Point(0, 0)
    outer_radius, inner_radius = 15.1, 15.0 - lane_width
    outer = center_point.buffer(outer_radius, quad_segs=64)
    inner = center_point.buffer(inner_radius, quad_segs=64)
    #walkable_area_poly = outer.difference(inner)
    walkable_area_poly = Polygon(outer.exterior.coords, holes=[inner.exterior.coords])
    walkable_area = WalkableArea(walkable_area_poly)
    
    r_max = 20
    thetas = np.linspace(0, 2*np.pi, num_sectors+1)
    measurement_areas = []
    for i in range(len(thetas)-1):
        theta1, theta2 = thetas[i], thetas[i+1]
        wedge = Polygon([(inner_radius * np.cos(theta1), inner_radius * np.sin(theta1)),
                         (inner_radius * np.cos(theta2), inner_radius * np.sin(theta2)),
                         (r_max * np.cos(theta2), r_max * np.sin(theta2)), 
                         (r_max * np.cos(theta1), r_max * np.sin(theta1))])
        # intersection to get measurement area as a convex set
        measurement_area_poly = outer.intersection(wedge)
        measurement_areas.append(MeasurementArea(measurement_area_poly))
    
    return walkable_area, measurement_areas


def compute_voronoi_states(traj, walkable_area, measurement_areas):
    try:
        individual = compute_individual_voronoi_polygons(
            traj_data=traj, walkable_area=walkable_area,
            cut_off=Cutoff(radius=(1.8+1.5)/2, quad_segments=1)
        )
    except IndexError:      
        plt.figure()
        plot_measurement_setup(
            walkable_area=walkable_area, traj=traj, traj_alpha=0.5, traj_width=1,
            measurement_areas=measurement_areas, ma_line_width=2, ma_alpha=0.5,
        ).set_aspect("equal")
        plt.show()
        sys.exit(1)
    
    individual_speed_single_sided = compute_individual_speed(
        traj_data=traj, frame_step=int(traj.frame_rate), compute_velocity=False,
        speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED,
    )
    individual_joined = individual_speed_single_sided.merge(individual, on=['id', 'frame'], how='inner')
    individual_joined['flow'] = individual_joined['density'] * individual_joined['speed'] # bic/m^2 * m/s = bic/s/m
    # switch flow and speed columns
    individual_joined['temp'] = individual_joined['flow']
    individual_joined['flow'] = individual_joined['speed']
    individual_joined['speed'] = individual_joined['temp']
    individual_joined = individual_joined.drop(columns=['temp'])
    
    voronoi_density_areas, voronoi_speed_areas = [], []
    for ma in measurement_areas:
        density_voronoi, intersecting = compute_voronoi_density(
            individual_voronoi_data=individual, measurement_area=ma
        )
        voronoi_density_areas.append(density_voronoi)
    
        voronoi_speed = compute_voronoi_speed(
            traj_data=traj, individual_voronoi_intersection=intersecting,
            individual_speed=individual_joined, measurement_area=ma,
        )
        voronoi_speed_areas.append(voronoi_speed)
    
    voronoi_states_areas = []
    for i in range(len(measurement_areas)):
        voronoi_states = voronoi_density_areas[i].join(voronoi_speed_areas[i], how="inner")
        voronoi_states = voronoi_states.rename(columns={'speed': 'flow'}) # switch back
        voronoi_states['density'] = voronoi_states['density']*1000.0 # bic/km/m
        voronoi_states['flow'] = voronoi_states['flow']*3600.0 # bic/h/m
        voronoi_states['speed'] = voronoi_states['flow'] / voronoi_states['density'] # bic/h/m / bic/km/m = km/h
        voronoi_states_areas.append(voronoi_states)
    
    voronoi_states_all = pd.concat(voronoi_states_areas)
    voronoi_states_all = voronoi_states_all.rename(columns={
        'density': 'Density', 'flow': 'Flow', 'speed': 'Speed'
    })
    return voronoi_states_all


# #############################################################################
# MAIN: Computing PFD Pseudo-States by Video
# #############################################################################
if COMPUTE_PSEUDO_STATES:    
    ts_df_all_X = None
    for video in CRB_Config.videos:
        print(f"... Processing video {video}.")
        walkable_area, measurement_areas = define_measurement_setup(NUM_SECTORS, CRB_Config.video_lane_widths[video])
        counter = 0
        for part in CRB_Config.video_parts_X[video]:
            filepath = f"../data/pedpy_traj/{video}_{part}.txt"
            if not os.path.exists(filepath):
                df = pd.read_csv(CRB_Config.data_root + f"{video}_{part}.txt")
                df = prepare_data_pedpy(df, CRB_Config.video_lane_widths[video], config=CRB_Config)
                df.to_csv(filepath, index=False, sep='\t', header=False)
            traj = load_trajectory(
                trajectory_file=pathlib.Path(filepath),
                default_frame_rate=25,
                default_unit='m'
            )
            ts_df = compute_voronoi_states(traj, walkable_area, measurement_areas)
            ts_df["Video"] = video
            ts_df["Video_Part"] = part
            
            if ts_df_all_X is None:
                ts_df_all_X = ts_df.copy()
            else:
                ts_df_all_X = pd.concat((ts_df_all_X, ts_df))
            
            del ts_df
            counter += 1
            print(f"... Processed video part {part}. Finished {counter}/{len(CRB_Config.video_parts_X[video])}.")
    gc.collect()    
    
    ts_df_all_X.to_csv("../data/CRB_Voronoi_PseudoTrafficStates_ALLVideos.txt", index=False)
else:
    ts_df_all_X = pd.read_csv("../data/CRB_Voronoi_PseudoTrafficStates_ALLVideos.txt")

# # These factors are the mean Polar_Y_Dist for each set of lane width settings
# ts_df_all_X.loc[ts_df_all_X['Video'].isin(CRB_Config.videos[:-3]), 'Density'] *= (0.426+0.6128)/2
# ts_df_all_X.loc[ts_df_all_X['Video'].isin(CRB_Config.videos[:-3]), 'Flow'] *= (0.426+0.6128)/2
# ts_df_all_X.loc[ts_df_all_X['Video'].isin(CRB_Config.videos[-3:]), 'Density'] *= (0.566+0.78)/2
# ts_df_all_X.loc[ts_df_all_X['Video'].isin(CRB_Config.videos[-3:]), 'Flow'] *= (0.566+0.78)/2

pfd_df_all_X = pd.read_csv("../data/CRB_PseudoTrafficStates_ALLVideos_V2.txt")
ts_df_all_SSD_X = pd.read_csv("../data/CRB_SSD_PseudoTrafficStates_ALLVideos.txt")

# #############################################################################
# MAIN: PFD Analysis by Videos with SAME lane width
# #############################################################################
# LANE WIDTH 2.5 METERS
video_set = CRB_Config.videos[:-3]
ts_agg_df, fig = aggregate_fd(ts_df_all_X[ts_df_all_X['Video'].isin(video_set)], 
                      max_density=200, bin_width=0.3, min_observations=MIN_OBS, 
                      FD_form="ExpFD", loss_fn="HuberLoss", 
                      show_pseudo_states=False, log_results=False)
plt.close(fig)
pfd_agg_df, fig = aggregate_fd(pfd_df_all_X[pfd_df_all_X['Video'].isin(video_set)], 
                      max_density=200, bin_width=0.3, min_observations=MIN_OBS, 
                      FD_form="ExpFD", loss_fn="HuberLoss", 
                      show_pseudo_states=False, log_results=False)
plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].scatter(pfd_agg_df["Density"], pfd_agg_df["Flow"], label="BFD (Proposed)", alpha=0.5)
axs[0].scatter(ts_agg_df["Density"], ts_agg_df["Flow"], label="Voronoi", alpha=0.5)
axs[0].set_xlabel("Density [bic/km/m]")
axs[0].set_ylabel("Flow [bic/h/m]")
axs[0].set_ylim([0, 2500])
axs[0].set_xlim([0, 200])

axs[1].scatter(pfd_agg_df["Density"], pfd_agg_df["Speed"], label="BFD (Proposed)", alpha=0.5)
axs[1].scatter(ts_agg_df["Density"], ts_agg_df["Speed"], label="Voronoi", alpha=0.5)
axs[1].set_xlabel("Density [bic/km/m]")
axs[1].set_ylabel("Speed [km/h]")
axs[1].set_ylim([0, 20])
axs[1].set_xlim([0, 200])

h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, bbox_to_anchor=(0.5, -0.05), loc='lower center', ncol=3, bbox_transform=fig.transFigure)
fig.tight_layout()
fig.savefig("../figures/BFD_Voronoi_Comparison_LaneWidth_2p5.pdf", dpi=300, bbox_inches='tight')

# LANE WIDTH 3.75 METERS
video_set = CRB_Config.videos[-3:]
ts_agg_df, fig = aggregate_fd(ts_df_all_X[ts_df_all_X['Video'].isin(video_set)], 
                      max_density=200, bin_width=0.3, min_observations=MIN_OBS, 
                      FD_form="ExpFD", loss_fn="HuberLoss", 
                      show_pseudo_states=False, log_results=False)
plt.close(fig)
pfd_agg_df, fig = aggregate_fd(pfd_df_all_X[pfd_df_all_X['Video'].isin(video_set)], 
                      max_density=200, bin_width=0.3, min_observations=MIN_OBS, 
                      FD_form="ExpFD", loss_fn="HuberLoss", 
                      show_pseudo_states=False, log_results=False)
plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].scatter(pfd_agg_df["Density"], pfd_agg_df["Flow"], label="BFD (Proposed)", alpha=0.5)
axs[0].scatter(ts_agg_df["Density"], ts_agg_df["Flow"], label="Voronoi", alpha=0.5)
axs[0].set_xlabel("Density [bic/km/m]")
axs[0].set_ylabel("Flow [bic/h/m]")
axs[0].set_ylim([0, 2500])
axs[0].set_xlim([0, 200])

axs[1].scatter(pfd_agg_df["Density"], pfd_agg_df["Speed"], label="BFD (Proposed)", alpha=0.5)
axs[1].scatter(ts_agg_df["Density"], ts_agg_df["Speed"], label="Voronoi", alpha=0.5)
axs[1].set_xlabel("Density [bic/km/m]")
axs[1].set_ylabel("Speed [km/h]")
axs[1].set_ylim([0, 20])
axs[1].set_xlim([0, 200])

h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, bbox_to_anchor=(0.5, -0.05), loc='lower center', ncol=3, bbox_transform=fig.transFigure)
fig.tight_layout()
fig.savefig("../figures/BFD_Voronoi_Comparison_LaneWidth_3p75.pdf", dpi=300, bbox_inches='tight')

plt.show()

# #############################################################################
# MAIN: PFD Analysis by Videos with SAME lane width, ALL 3 METHODS
# #############################################################################
# LANE WIDTH 2.5 METERS
video_set = CRB_Config.videos[:-3]
tsVor_agg_df, fig = aggregate_fd(ts_df_all_X[ts_df_all_X['Video'].isin(video_set)], 
                                 max_density=200, bin_width=0.3, min_observations=MIN_OBS, 
                                 FD_form="ExpFD", loss_fn="HuberLoss", 
                                 show_pseudo_states=False, log_results=False)
plt.close(fig)
tsSSD_agg_df, fig = aggregate_fd(
    ts_df_all_SSD_X[(ts_df_all_SSD_X['Video'].isin(video_set)) & (ts_df_all_SSD_X["Num_Bicycles"] >= MIN_BIKES)], 
    max_density=200, bin_width=0.3, min_observations=MIN_OBS, FD_form="ExpFD", 
    loss_fn="HuberLoss", show_pseudo_states=False, log_results=False)
plt.close(fig)
tsSSD_agg_df = tsSSD_agg_df[tsSSD_agg_df['Num_Observations'] >= 3]
pfd_agg_df, fig = aggregate_fd(pfd_df_all_X[pfd_df_all_X['Video'].isin(video_set)], 
                      max_density=200, bin_width=0.3, min_observations=MIN_OBS, 
                      FD_form="ExpFD", loss_fn="HuberLoss", 
                      show_pseudo_states=False, log_results=False)
plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].scatter(pfd_agg_df["Density"], pfd_agg_df["Flow"], label="BFD (Proposed)", alpha=0.5)
axs[0].scatter(tsSSD_agg_df["Density"], tsSSD_agg_df["Flow"], label="SSD", alpha=0.5)
axs[0].scatter(tsVor_agg_df["Density"], tsVor_agg_df["Flow"], label="Voronoi", alpha=0.5)
axs[0].set_xlabel("Density [bic/km/m]")
axs[0].set_ylabel("Flow [bic/h/m]")
axs[0].set_ylim([0, 2000])
axs[0].set_xlim([0, 200])

axs[1].scatter(pfd_agg_df["Density"], pfd_agg_df["Speed"], label="BFD (Proposed)", alpha=0.5)
axs[1].scatter(tsSSD_agg_df["Density"], tsSSD_agg_df["Speed"], label="SSD", alpha=0.5)
axs[1].scatter(tsVor_agg_df["Density"], tsVor_agg_df["Speed"], label="Voronoi", alpha=0.5)
axs[1].set_xlabel("Density [bic/km/m]")
axs[1].set_ylabel("Speed [km/h]")
axs[1].set_ylim([0, 20])
axs[1].set_xlim([0, 200])

h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, bbox_to_anchor=(0.5, -0.05), loc='lower center', ncol=3, bbox_transform=fig.transFigure)
fig.tight_layout()
fig.savefig("../figures/BFD_SSD_Voronoi_Comparison_LaneWidth_2p5.pdf", dpi=300, bbox_inches='tight')


# LANE WIDTH 3.75 METERS
video_set = CRB_Config.videos[-3:]
tsVor_agg_df, fig = aggregate_fd(ts_df_all_X[ts_df_all_X['Video'].isin(video_set)], 
                                 max_density=200, bin_width=0.3, min_observations=MIN_OBS, 
                                 FD_form="ExpFD", loss_fn="HuberLoss", 
                                 show_pseudo_states=False, log_results=False)
plt.close(fig)
tsSSD_agg_df, fig = aggregate_fd(
    ts_df_all_SSD_X[(ts_df_all_SSD_X['Video'].isin(video_set)) & (ts_df_all_SSD_X["Num_Bicycles"] >= MIN_BIKES)], 
    max_density=200, bin_width=0.3, min_observations=MIN_OBS, FD_form="ExpFD", 
    loss_fn="HuberLoss", show_pseudo_states=False, log_results=False)
plt.close(fig)
tsSSD_agg_df = tsSSD_agg_df[tsSSD_agg_df['Num_Observations'] >= 3]
pfd_agg_df, fig = aggregate_fd(pfd_df_all_X[pfd_df_all_X['Video'].isin(video_set)], 
                      max_density=200, bin_width=0.3, min_observations=MIN_OBS, 
                      FD_form="ExpFD", loss_fn="HuberLoss", 
                      show_pseudo_states=False, log_results=False)
plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].scatter(pfd_agg_df["Density"], pfd_agg_df["Flow"], label="BFD (Proposed)", alpha=0.5)
axs[0].scatter(tsSSD_agg_df["Density"], tsSSD_agg_df["Flow"], label="SSD", alpha=0.5)
axs[0].scatter(tsVor_agg_df["Density"], tsVor_agg_df["Flow"], label="Voronoi", alpha=0.5)
axs[0].set_xlabel("Density [bic/km/m]")
axs[0].set_ylabel("Flow [bic/h/m]")
axs[0].set_ylim([0, 2000])
axs[0].set_xlim([0, 200])

axs[1].scatter(pfd_agg_df["Density"], pfd_agg_df["Speed"], label="BFD (Proposed)", alpha=0.5)
axs[1].scatter(tsSSD_agg_df["Density"], tsSSD_agg_df["Speed"], label="SSD", alpha=0.5)
axs[1].scatter(tsVor_agg_df["Density"], tsVor_agg_df["Speed"], label="Voronoi", alpha=0.5)
axs[1].set_xlabel("Density [bic/km/m]")
axs[1].set_ylabel("Speed [km/h]")
axs[1].set_ylim([0, 20])
axs[1].set_xlim([0, 200])

h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, bbox_to_anchor=(0.5, -0.05), loc='lower center', ncol=3, bbox_transform=fig.transFigure)
fig.tight_layout()
fig.savefig("../figures/BFD_SSD_Voronoi_Comparison_LaneWidth_3p75.pdf", dpi=300, bbox_inches='tight')

plt.show()
