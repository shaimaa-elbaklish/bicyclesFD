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
import random
import warnings
warnings.simplefilter('ignore', RuntimeWarning) # Ignore all RuntimeWarnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _log_config import create_log_file
create_log_file(logfile = "../logs/CRB_FD_Validation.log")
from _log_config import logger, enable_logging_overwrite, enable_logging_append
enable_logging_overwrite()

from _constants import CRB_Config
from tools_data import compute_lane_coordinates
from tools_fd import compute_pseudo_states_ssd
from tools_bfd import aggregate_fd

# #############################################################################
# CONSTANTS
# #############################################################################
CRB_Config = CRB_Config()
COMPUTE_PSEUDO_STATES = False

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

# for static spatial discretization method
DT = 4.0
NUM_SECTORS = 5
MIN_BIKES = 2

# #############################################################################
# MAIN: Computing PFD Pseudo-States by Video
# #############################################################################
if COMPUTE_PSEUDO_STATES:
    ts_df_all_X = None
    for video in CRB_Config.videos:
        print(f"... Processing video {video}.")
        counter = 0
        for part in CRB_Config.video_parts_X[video]:
            df = pd.read_csv(CRB_Config.data_root + f"{video}_{part}.txt")
            df = compute_lane_coordinates(df)
            ts_df = compute_pseudo_states_ssd(df, num_sectors=NUM_SECTORS, dt=DT, 
                                              lane_width=CRB_Config.video_lane_widths[video],
                                              config=CRB_Config, TTD_USE_POLAR=True)
            ts_df["Density"] = ts_df["Density"]/CRB_Config.video_lane_widths[video]
            ts_df["Flow"] = ts_df["Flow"]/CRB_Config.video_lane_widths[video]
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
    
    ts_df_all_X.to_csv("../data/CRB_SSD_PseudoTrafficStates_ALLVideos.txt", index=False)
else:
    ts_df_all_X = pd.read_csv("../data/CRB_SSD_PseudoTrafficStates_ALLVideos.txt")

pfd_df_all_X = pd.read_csv("../data/CRB_PseudoTrafficStates_ALLVideos.txt")

# #############################################################################
# MAIN: PFD Analysis by Videos with SAME lane width
# #############################################################################
# LANE WIDTH 2.5 METERS
ts_agg_df, fig = aggregate_fd(ts_df_all_X[(ts_df_all_X['Video'].isin(CRB_Config.videos[:-3])) & (ts_df_all_X["Num_Bicycles"] >= MIN_BIKES)], 
                      max_density=200, bin_width=0.3, min_observations=50, 
                      FD_form="ExpFD", loss_fn="HuberLoss", 
                      show_pseudo_states=False, log_results=False)
plt.close(fig)
# min=1,max=10, mean=2.13, median=2
# force min_observation to be 3 at least
ts_agg_df = ts_agg_df[ts_agg_df['Num_Observations'] >= 3]
pfd_agg_df, fig = aggregate_fd(pfd_df_all_X[pfd_df_all_X['Video'].isin(CRB_Config.videos[:-3])], 
                      max_density=200, bin_width=0.3, min_observations=50, 
                      FD_form="ExpFD", loss_fn="HuberLoss", 
                      show_pseudo_states=False, log_results=False)
plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].scatter(pfd_agg_df["Density"], pfd_agg_df["Flow"], label="BFD (Proposed)", alpha=0.5)
axs[0].scatter(ts_agg_df["Density"], ts_agg_df["Flow"], label="SSD", alpha=0.5)
axs[0].set_xlabel("Density [bic/km/m]")
axs[0].set_ylabel("Flow [bic/h/m]")
axs[0].set_ylim([0, 2500])
axs[0].set_xlim([0, 200])

axs[1].scatter(pfd_agg_df["Density"], pfd_agg_df["Speed"], label="BFD (Proposed)", alpha=0.5)
axs[1].scatter(ts_agg_df["Density"], ts_agg_df["Speed"], label="SSD", alpha=0.5)
axs[1].set_xlabel("Density [bic/km/m]")
axs[1].set_ylabel("Speed [km/h]")
axs[1].set_ylim([0, 20])
axs[1].set_xlim([0, 200])

h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, bbox_to_anchor=(0.5, -0.05), loc='lower center', ncol=3, bbox_transform=fig.transFigure)
fig.tight_layout()
fig.savefig("../figures/BFD_SSD_Comparison_LaneWidth_2p5.png", dpi=300, bbox_inches='tight')

# LANE WIDTH 3.75 METERS
ts_agg_df, fig = aggregate_fd(ts_df_all_X[(ts_df_all_X['Video'].isin(CRB_Config.videos[-3:])) & (ts_df_all_X["Num_Bicycles"] >= MIN_BIKES)], 
                      max_density=200, bin_width=0.3, min_observations=50, 
                      FD_form="ExpFD", loss_fn="HuberLoss", 
                      show_pseudo_states=False, log_results=False)
plt.close(fig)
# min=1,max=8, mean=1.79, median=1
# force min_observation to be 3 at least
ts_agg_df = ts_agg_df[ts_agg_df['Num_Observations'] >= 3]
pfd_agg_df, fig = aggregate_fd(pfd_df_all_X[pfd_df_all_X['Video'].isin(CRB_Config.videos[-3:])], 
                      max_density=200, bin_width=0.3, min_observations=50, 
                      FD_form="ExpFD", loss_fn="HuberLoss", 
                      show_pseudo_states=False, log_results=False)
plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].scatter(pfd_agg_df["Density"], pfd_agg_df["Flow"], label="BFD (Proposed)", alpha=0.5)
axs[0].scatter(ts_agg_df["Density"], ts_agg_df["Flow"], label="SSD", alpha=0.5)
axs[0].set_xlabel("Density [bic/km/m]")
axs[0].set_ylabel("Flow [bic/h/m]")
axs[0].set_ylim([0, 2500])
axs[0].set_xlim([0, 200])

axs[1].scatter(pfd_agg_df["Density"], pfd_agg_df["Speed"], label="BFD (Proposed)", alpha=0.5)
axs[1].scatter(ts_agg_df["Density"], ts_agg_df["Speed"], label="SSD", alpha=0.5)
axs[1].set_xlabel("Density [bic/km/m]")
axs[1].set_ylabel("Speed [km/h]")
axs[1].set_ylim([0, 20])
axs[1].set_xlim([0, 200])

h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, bbox_to_anchor=(0.5, -0.05), loc='lower center', ncol=3, bbox_transform=fig.transFigure)
fig.tight_layout()
fig.savefig("../figures/BFD_SSD_Comparison_LaneWidth_3p75.png", dpi=300, bbox_inches='tight')

plt.show()