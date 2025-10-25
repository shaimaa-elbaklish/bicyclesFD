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

from pedpy import load_trajectory

from _log_config import create_log_file
create_log_file(logfile = "../logs/CRB_FD_Validation.log")
from _log_config import logger, enable_logging_overwrite, enable_logging_append
enable_logging_overwrite()

from _constants import CRB_Config
from tools_voronoi import prepare_data_pedpy, define_measurement_setup
from tools_voronoi import compute_voronoi_states
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
