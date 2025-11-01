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
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _log_config import create_log_file
create_log_file(logfile = "../logs/CRB_FD_delta_sensitivity.log")
from _log_config import logger, enable_logging_overwrite
enable_logging_overwrite()

from _constants import CRB_Config
from tools_bfd import aggregate_fd

# #############################################################################
# CONSTANTS
# #############################################################################
CRB_Config = CRB_Config()

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

DELTA = 0.3
MIN_OBS = 50

bin_width_grid = [
    0.05, 0.1, 0.3, 0.6,
    0.8, 1, 1.5, 2
]

min_obs_grid = [
    15, 25, 40, 50,
    80, 100, 120, 150
]

# #############################################################################
# METHODS
# #############################################################################
def redraw_fd(fd_fig, draw_ax, max_density, scatter_label, scatter_color, line_color):
    axs = fd_fig.get_axes()

    line_free_pfd = axs[0].lines[0].get_xydata()
    line_cong_pfd = axs[0].lines[1].get_xydata()
    scatter_pfd = axs[0].collections[0].get_offsets()
    
    draw_ax.scatter(scatter_pfd[:, 0], scatter_pfd[:, 1], alpha=0.2, label=scatter_label, color=scatter_color)
    draw_ax.plot(line_free_pfd[:, 0], line_free_pfd[:, 1], color=line_color, linestyle="dashed")
    draw_ax.plot(line_cong_pfd[:, 0], line_cong_pfd[:, 1], color=line_color, linestyle="dashed")
    draw_ax.set(xlim=[0, max_density], ylim=[0, 1600])
    density_ticks = np.linspace(0, max_density, 5)
    flow_ticks = np.linspace(0, 1600, 5)
    draw_ax.set_xticks(density_ticks)
    draw_ax.set_yticks(flow_ticks)
    return draw_ax

# #############################################################################
# MAIN: Sensitivity Analysis on Bin Width
# #############################################################################
pfd_df_all_X = pd.read_csv("../data/CRB_PseudoTrafficStates_ALLVideos_V2.txt") 

video_set_2p5 = CRB_Config.videos[:-3]
video_set_3p75 = CRB_Config.videos[-3:]

num_cols, num_rows = 4, 2
mainfig, mainaxs = plt.subplots(num_rows, num_cols, figsize=(num_cols*3, num_rows*3), sharex=True, sharey=True)

figrow = 0
figcol = 0
for delta in bin_width_grid:
    random.seed(seed_value)
    np.random.seed(seed_value)
    logger.info(f"DENSITY BIN WIDTH = {delta:2f} bic/km/m")
    
    
    logger.info("LANE WIDTH 2.5 METERS")
    # Estimate Jam Density
    tmp_df = pfd_df_all_X[pfd_df_all_X['Video'].isin(video_set_2p5)]
    tmp_df['v_Vel'] = tmp_df['Space_Hdwy'] / tmp_df['Time_Hdwy']
    tmp_df = tmp_df[tmp_df['v_Vel'].abs() <= 0.1]
    jam_density_est = 1000/tmp_df['Space_Hdwy'].median() / 2.5
    # Do the Curve Fitting
    _, fig = aggregate_fd(pfd_df_all_X[pfd_df_all_X['Video'].isin(video_set_2p5)], 
                          max_density=200, bin_width=delta, min_observations=MIN_OBS, 
                          FD_form="WuFD", loss_fn="HuberLoss", jam_density=jam_density_est,
                          show_pseudo_states=False, log_results=True,
                          k_cong_ratio=0.85)
    redraw_fd(fig, mainaxs[figrow, figcol], max_density=200, scatter_label='$lw$ = 2.5m',
              scatter_color='tab:blue', line_color='blue')
    plt.close(fig)
    
    
    logger.info("LANE WIDTH 3.75 METERS")
    # Estimate Jam Density
    tmp_df = pfd_df_all_X[pfd_df_all_X['Video'].isin(video_set_3p75)]
    tmp_df['v_Vel'] = tmp_df['Space_Hdwy'] / tmp_df['Time_Hdwy']
    tmp_df = tmp_df[tmp_df['v_Vel'].abs() <= 0.1]
    jam_density_est = 1000/tmp_df['Space_Hdwy'].median() / 3.75
    # Do the Curve Fitting
    _, fig = aggregate_fd(pfd_df_all_X[pfd_df_all_X['Video'].isin(video_set_3p75)], 
                          max_density=200, bin_width=delta, min_observations=MIN_OBS, 
                          FD_form="WuFD", loss_fn="HuberLoss", jam_density=jam_density_est,
                          show_pseudo_states=False, log_results=True,
                          k_cong_ratio=0.85)
    redraw_fd(fig, mainaxs[figrow, figcol], max_density=200, scatter_label='$lw$ = 3.75m',
              scatter_color='tab:red', line_color='red')
    plt.close(fig)
    
    mainaxs[figrow, figcol].set_title(f"$\delta_\kappa$ = {delta} bic/km/m")
    
    if figrow == 1:
        mainaxs[figrow, figcol].set_xlabel('Density [bic/km/m]')
    if figcol == 0:
        mainaxs[figrow, figcol].set_ylabel('Flow [bic/h/m]')
    
    figcol += 1
    if figcol >= num_cols:
        figcol = 0
        figrow += 1

mainaxs[0, 0].legend()
mainfig.tight_layout()

mainfig.savefig('../figures/BFD_Bin_Width_Sensitivity_Analysis.pdf', dpi=300, bbox_inches='tight')


# #############################################################################
# MAIN: Sensitivity Analysis on Min Observations
# #############################################################################
from _log_config import create_log_file
create_log_file(logfile = "../logs/CRB_FD_min_obs_sensitivity.log")
from _log_config import logger, enable_logging_overwrite
enable_logging_overwrite()

num_cols, num_rows = 4, 2
mainfig, mainaxs = plt.subplots(num_rows, num_cols, figsize=(num_cols*3, num_rows*3), sharex=True, sharey=True)

figrow = 0
figcol = 0
for min_obs in min_obs_grid:
    random.seed(seed_value)
    np.random.seed(seed_value)
    logger.info(f"MIN. OBSERVATIONS = {min_obs}")
    
    
    logger.info("LANE WIDTH 2.5 METERS")
    # Estimate Jam Density
    tmp_df = pfd_df_all_X[pfd_df_all_X['Video'].isin(video_set_2p5)]
    tmp_df['v_Vel'] = tmp_df['Space_Hdwy'] / tmp_df['Time_Hdwy']
    tmp_df = tmp_df[tmp_df['v_Vel'].abs() <= 0.1]
    jam_density_est = 1000/tmp_df['Space_Hdwy'].median() / 2.5
    # Do the Curve Fitting
    _, fig = aggregate_fd(pfd_df_all_X[pfd_df_all_X['Video'].isin(video_set_2p5)], 
                          max_density=200, bin_width=DELTA, min_observations=min_obs, 
                          FD_form="WuFD", loss_fn="HuberLoss", jam_density=jam_density_est,
                          show_pseudo_states=False, log_results=True,
                          k_cong_ratio=0.85)
    redraw_fd(fig, mainaxs[figrow, figcol], max_density=200, scatter_label='$lw$ = 2.5m',
              scatter_color='tab:blue', line_color='blue')
    plt.close(fig)
    
    
    logger.info("LANE WIDTH 3.75 METERS")
    # Estimate Jam Density
    tmp_df = pfd_df_all_X[pfd_df_all_X['Video'].isin(video_set_3p75)]
    tmp_df['v_Vel'] = tmp_df['Space_Hdwy'] / tmp_df['Time_Hdwy']
    tmp_df = tmp_df[tmp_df['v_Vel'].abs() <= 0.1]
    jam_density_est = 1000/tmp_df['Space_Hdwy'].median() / 3.75
    # Do the Curve Fitting
    _, fig = aggregate_fd(pfd_df_all_X[pfd_df_all_X['Video'].isin(video_set_3p75)], 
                          max_density=200, bin_width=DELTA, min_observations=min_obs, 
                          FD_form="WuFD", loss_fn="HuberLoss", jam_density=jam_density_est,
                          show_pseudo_states=False, log_results=True,
                          k_cong_ratio=0.85)
    redraw_fd(fig, mainaxs[figrow, figcol], max_density=200, scatter_label='$lw$ = 3.75m',
              scatter_color='tab:red', line_color='red')
    plt.close(fig)
    
    mainaxs[figrow, figcol].set_title(f"$N_{{o, min}}$ = {min_obs}")
    
    if figrow == 1:
        mainaxs[figrow, figcol].set_xlabel('Density [bic/km/m]')
    if figcol == 0:
        mainaxs[figrow, figcol].set_ylabel('Flow [bic/h/m]')
    
    figcol += 1
    if figcol >= num_cols:
        figcol = 0
        figrow += 1

mainaxs[0, 0].legend()
mainfig.tight_layout()

mainfig.savefig('../figures/BFD_Min_Observations_Sensitivity_Analysis.pdf', dpi=300, bbox_inches='tight')