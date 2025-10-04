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
warnings.simplefilter('ignore', UserWarning) # Ignore all UserWarning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CONSTANT for BFD
USE_HEADWAY_HOOGENDOORN = False
from _log_config import create_log_file
if USE_HEADWAY_HOOGENDOORN:
    create_log_file(logfile = "../logs/CRB_FD_Analysis_Hoogendoorn.log")
else:
    create_log_file(logfile = "../logs/CRB_FD_Analysis.log")
from _log_config import logger, enable_logging_overwrite
enable_logging_overwrite()

from _constants import CRB_Config
from tools_data import compute_lane_coordinates
from tools_data import determine_leader, determine_leader_V2, determine_leader_Hoogendoorn
from tools_bfd import compute_pseudo_states_pfd_N2
from tools_bfd import aggregate_fd

# #############################################################################
# CONSTANTS
# #############################################################################
CRB_Config = CRB_Config()
COMPUTE_PSEUDO_STATES = False

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

# for aggregation
MIN_OBS = 50

# #############################################################################
# MAIN: Computing PFD Pseudo-States by Video
# #############################################################################
if COMPUTE_PSEUDO_STATES:
    pfd_df_all_X = None
    for video in CRB_Config.videos:
        print(f"... Processing video {video}.")
        counter = 0
        for part in CRB_Config.video_parts_X[video]:
            df = pd.read_csv(CRB_Config.data_root + f"{video}_{part}.txt")
            df = compute_lane_coordinates(df)
            if USE_HEADWAY_HOOGENDOORN:
                df = determine_leader_Hoogendoorn(df)
            else:
                df = determine_leader_V2(df)
            pfd_df = compute_pseudo_states_pfd_N2(df, CRB_Config.video_lane_widths[video], config=CRB_Config)
            pfd_df["Density"] = pfd_df["Density"]/CRB_Config.video_lane_widths[video]
            pfd_df["Flow"] = pfd_df["Flow"]/CRB_Config.video_lane_widths[video]
            pfd_df["Video"] = video
            pfd_df["Video_Part"] = part
            
            if pfd_df_all_X is None:
                pfd_df_all_X = pfd_df.copy()
            else:
                pfd_df_all_X = pd.concat((pfd_df_all_X, pfd_df))
            
            del pfd_df
            counter += 1
            print(f"... Processed video part {part}. Finished {counter}/{len(CRB_Config.video_parts_X[video])}.")
    gc.collect()    
    
    if USE_HEADWAY_HOOGENDOORN:
        pfd_df_all_X.to_csv("../data/CRB_PseudoTrafficStates_Hoogendoorn_ALLVideos_V2.txt", index=False)
    else:
        pfd_df_all_X.to_csv("../data/CRB_PseudoTrafficStates_ALLVideos_V2.txt", index=False) 
else:
    if USE_HEADWAY_HOOGENDOORN:
        pfd_df_all_X = pd.read_csv("../data/CRB_PseudoTrafficStates_Hoogendoorn_ALLVideos_V2.txt")
    else:
        pfd_df_all_X = pd.read_csv("../data/CRB_PseudoTrafficStates_ALLVideos_V2.txt") 
# sys.exit(1)

# #############################################################################
# MAIN: PFD Analysis by Videos with SAME lane width
# #############################################################################
logger.info("... NOW, PROCESSING ALL VIDEOS WITH LANE WIDTH = 2.5 METERS")
video_set = CRB_Config.videos[:-3]
# Estimate Jam Denisty
if USE_HEADWAY_HOOGENDOORN:
    jam_density_est = None
else:
    tmp_df = pfd_df_all_X[pfd_df_all_X['Video'].isin(video_set)]
    tmp_df['v_Vel'] = tmp_df['Space_Hdwy'] / tmp_df['Time_Hdwy']
    tmp_df = tmp_df[(tmp_df['v_Vel'] - 0.1).abs() <= 1e-02]
    jam_density_est = 1000/tmp_df['Space_Hdwy'].median() / 2.5

logger.info("BFD (i.e. Proposed Method)")
_, fig = aggregate_fd(pfd_df_all_X[pfd_df_all_X['Video'].isin(video_set)], 
                      max_density=200, bin_width=0.3, min_observations=MIN_OBS, 
                      FD_form="ExpFD", loss_fn="HuberLoss", 
                      show_pseudo_states=False, log_results=True)
if USE_HEADWAY_HOOGENDOORN:
    fig.savefig("../figures/BFD_Hoogendoorn_ExpFD_LaneWidth_2p5.pdf", dpi=300, bbox_inches='tight')
else:
    fig.savefig("../figures/BFD_ExpFD_LaneWidth_2p5.pdf", dpi=300, bbox_inches='tight')
fig.suptitle("[BFD] Videos with Lane Width = 2.5 m - Exponential FD")
fig.tight_layout()
# plt.close(fig)
_, fig = aggregate_fd(pfd_df_all_X[pfd_df_all_X['Video'].isin(video_set)], 
                      max_density=200, bin_width=0.3, min_observations=MIN_OBS, 
                      FD_form="WuFD", loss_fn="HuberLoss", 
                      show_pseudo_states=False, log_results=True, jam_density=jam_density_est)
if USE_HEADWAY_HOOGENDOORN:
    fig.savefig("../figures/BFD_Hoogendoorn_WuFD_LaneWidth_2p5.pdf", dpi=300, bbox_inches='tight')
else:
    fig.savefig("../figures/BFD_WuFD_LaneWidth_2p5.pdf", dpi=300, bbox_inches='tight')
fig.suptitle("[BFD] Videos with Lane Width = 2.5 m - Wu's FD")
fig.tight_layout()
# fig.savefig("../figures/BFD_WuFD_LaneWidth_2p5.png", dpi=300, bbox_inches='tight')
# plt.close(fig)



logger.info("... NOW, PROCESSING ALL VIDEOS WITH LANE WIDTH = 3.75 METERS")
video_set = CRB_Config.videos[-3:]
# Estimate Jam Density
if USE_HEADWAY_HOOGENDOORN:
    jam_density_est = None
else:
    tmp_df = pfd_df_all_X[pfd_df_all_X['Video'].isin(video_set)]
    tmp_df['v_Vel'] = tmp_df['Space_Hdwy'] / tmp_df['Time_Hdwy']
    tmp_df = tmp_df[(tmp_df['v_Vel'] - 0.1).abs() <= 1e-02]
    jam_density_est = 1000/tmp_df['Space_Hdwy'].median() / 3.75

logger.info("BFD (i.e. Proposed Method)")
_, fig = aggregate_fd(pfd_df_all_X[pfd_df_all_X['Video'].isin(video_set)], 
                      max_density=200, bin_width=0.3, min_observations=MIN_OBS, 
                      FD_form="ExpFD", loss_fn="HuberLoss", 
                      show_pseudo_states=False, log_results=True)
if USE_HEADWAY_HOOGENDOORN:
    fig.savefig("../figures/BFD_Hoogendoorn_ExpFD_LaneWidth_3p75.pdf", dpi=300, bbox_inches='tight')
else:
    fig.savefig("../figures/BFD_ExpFD_LaneWidth_3p75.pdf", dpi=300, bbox_inches='tight')
fig.suptitle("[BFD] Videos with Lane Width = 3.75 m - Exponential FD")
fig.tight_layout()
# plt.close(fig)
_, fig = aggregate_fd(pfd_df_all_X[pfd_df_all_X['Video'].isin(video_set)], 
                      max_density=500, bin_width=0.3, min_observations=MIN_OBS, 
                      FD_form="WuFD", loss_fn="HuberLoss", 
                      show_pseudo_states=False, log_results=True, jam_density=jam_density_est)
if USE_HEADWAY_HOOGENDOORN:
    fig.savefig("../figures/BFD_Hoogendoorn_WuFD_LaneWidth_3p75.pdf", dpi=300, bbox_inches='tight')
else:
    fig.savefig("../figures/BFD_WuFD_LaneWidth_3p75.pdf", dpi=300, bbox_inches='tight')
fig.suptitle("[BFD] Videos with Lane Width = 3.75 m - Wu's FD")
fig.tight_layout()
# fig.savefig("../figures/BFD_WuFD_LaneWidth_3p75.png", dpi=300, bbox_inches='tight')
# plt.close(fig)

plt.show()
