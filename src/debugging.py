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

from _log_config import create_log_file
create_log_file(logfile = "../logs/CRB_FD_Analysis.log")
from _log_config import logger, enable_logging_overwrite, enable_logging_append
enable_logging_overwrite()

from _constants import CRB_Config
from tools_data import compute_lane_coordinates
from tools_data import determine_leader, determine_leader_Hoogendoorn
from tools_bfd import compute_pseudo_states_pfd_N2
from tools_bfd import aggregate_fd

# #############################################################################
# CONSTANTS
# #############################################################################
CRB_Config = CRB_Config()

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

# video = CRB_Config.videos[-2]
# # part = CRB_Config.video_parts_X[video][1]

# ts_df_all_Vor = pd.read_csv("../data/CRB_Voronoi_PseudoTrafficStates_ALLVideos.txt")
# ts_df_all_Vor = ts_df_all_Vor[ts_df_all_Vor['Video'] == video]
# ts_df_all_SSD = pd.read_csv("../data/CRB_SSD_PseudoTrafficStates_ALLVideos.txt")
# ts_df_all_SSD = ts_df_all_SSD[ts_df_all_SSD['Video'] == video]

# #############################################################################
# MAIN: Comparing BFD methods
# #############################################################################

pfd_df_all = pd.read_csv("../data/CRB_PseudoTrafficStates_ALLVideos_V2.txt")
# pfd_df_all_Hoogendoorn = pd.read_csv("C:/Users/ShaimaaElBaklish/Documents/GitHub/bicycle_dataset/src/logs/PFD_X_PseudoStates_ALLVideos.txt")
pfd_df_all_Hoogendoorn = pd.read_csv("../data/CRB_PseudoTrafficStates_Hoogendoorn_ALLVideos_V2.txt")
ts_df_all = pd.read_csv("../data/CRB_Voronoi_PseudoTrafficStates_ALLVideos.txt")

# pfd_df_all = pd.read_csv("../data/CRB_PseudoTrafficStates_ALLVideos.txt")
# plt.figure()
# pfd_df_all.loc[pfd_df_all['Video'].isin(CRB_Config.videos[:-3]), 'Polar_Y_Dist'].hist(bins=1000, density=True, alpha=0.5)
# # mean=0.6128, median=0.4261
# pfd_df_all.loc[pfd_df_all['Video'].isin(CRB_Config.videos[-3:]), 'Polar_Y_Dist'].hist(bins=1000, density=True, alpha=0.5)
# # mean=0.7796, median=0.5660
# plt.legend(['2.5m', '3.75m'])

video_set = CRB_Config.videos[-3:]
bin_width = 0.3
pfd_df_all = pfd_df_all[pfd_df_all['Video'].isin(video_set)]
pfd_df_all['Density_Bin'] = pd.cut(pfd_df_all['Density'], bins=np.arange(0, 200.0, bin_width))
pfd_df_agg = pfd_df_all.groupby(["Density_Bin"], observed=False).agg({
    "Density": "mean", 
    "Flow": "mean",
    "Speed": "mean",
    "Density_Bin": "count"
})
pfd_df_agg = pfd_df_agg.rename(
    columns={"Density_Bin": "Num_Observations"}
)
pfd_df_agg = pfd_df_agg.dropna()
pfd_df_agg = pfd_df_agg[pfd_df_agg['Num_Observations'] >= 50]

pfd_df_all_Hoogendoorn = pfd_df_all_Hoogendoorn[pfd_df_all_Hoogendoorn['Video'].isin(video_set)]
pfd_df_all_Hoogendoorn['Density_Bin'] = pd.cut(pfd_df_all_Hoogendoorn['Density'], bins=np.arange(0, 200.0, bin_width))
pfd_df_Hoogendoorn_agg = pfd_df_all_Hoogendoorn.groupby(["Density_Bin"], observed=False).agg({
    "Density": "mean", 
    "Flow": "mean",
    "Speed": "mean",
    "Density_Bin": "count"
})
pfd_df_Hoogendoorn_agg = pfd_df_Hoogendoorn_agg.rename(
    columns={"Density_Bin": "Num_Observations"}
)
pfd_df_Hoogendoorn_agg = pfd_df_Hoogendoorn_agg.dropna()
pfd_df_Hoogendoorn_agg = pfd_df_Hoogendoorn_agg[pfd_df_Hoogendoorn_agg['Num_Observations'] >= 50]

ts_df_all = ts_df_all[ts_df_all['Video'].isin(video_set)]
ts_df_all['Density_Bin'] = pd.cut(ts_df_all['Density'], bins=np.arange(0, 200.0, bin_width))
ts_df_agg = ts_df_all.groupby(["Density_Bin"], observed=False).agg({
    "Density": "mean", 
    "Flow": "mean",
    "Speed": "mean",
    "Density_Bin": "count"
})
ts_df_agg = ts_df_agg.rename(
    columns={"Density_Bin": "Num_Observations"}
)
ts_df_agg = ts_df_agg.dropna()
ts_df_agg = ts_df_agg[ts_df_agg['Num_Observations'] >= 50]

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].scatter(ts_df_agg['Density'], ts_df_agg['Flow'], label='Voronoi', alpha=0.5)
axs[0].scatter(pfd_df_agg['Density'], pfd_df_agg['Flow'], label='BFD, $\Delta x^P_{max} = 60\deg$ (Proposed)', alpha=0.5)
axs[0].scatter(pfd_df_Hoogendoorn_agg['Density'], pfd_df_Hoogendoorn_agg['Flow'], label='BFD, $\Delta x^P_{max} = 360\deg$', alpha=0.25)
axs[0].set_ylim([0, 2000])
axs[0].set_xlim([0, 200])
axs[0].set_xlabel("Density [bic/km/m]")
axs[0].set_ylabel("Flow [bic/h/m]")

axs[1].scatter(ts_df_agg['Density'], ts_df_agg['Speed'], label='Voronoi', alpha=0.5)
axs[1].scatter(pfd_df_agg['Density'], pfd_df_agg['Speed'], label='BFD, $\Delta x^P_{max} = 60\deg$ (Proposed)', alpha=0.5)
axs[1].scatter(pfd_df_Hoogendoorn_agg['Density'], pfd_df_Hoogendoorn_agg['Speed'], label='BFD, $\Delta x^P_{max} = 360\deg$', alpha=0.25)
axs[1].set_ylim([0, 20])
axs[1].set_xlim([0, 200])
axs[1].set_xlabel("Density [bic/km/m]")
axs[1].set_ylabel("Speed [km/h]")


h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, bbox_to_anchor=(0.5, -0.07), loc='lower center', ncol=3, bbox_transform=fig.transFigure)
fig.tight_layout()
# fig.savefig("../figures/BFD_Hoogendoorn_Voronoi_Comparison_LaneWidth_2p5.pdf", dpi=300, bbox_inches='tight')
fig.savefig("../figures/BFD_Hoogendoorn_Voronoi_Comparison_LaneWidth_3p75.pdf", dpi=300, bbox_inches='tight')

sys.exit(1)

# #############################################################################
# # MAIN: Computing PFD Pseudo-States by Video
# # #############################################################################
# pfd_df_all = None
# print(f"... Processing video {video}.")
# counter = 0
# for part in CRB_Config.video_parts_X[video]:
#     df = pd.read_csv(CRB_Config.data_root + f"{video}_{part}.txt")
#     df = compute_lane_coordinates(df)
#     df = determine_leader(df)
#     pfd_df = compute_pseudo_states_pfd_N2(df, CRB_Config.video_lane_widths[video], config=CRB_Config)
#     pfd_df["Density"] = pfd_df["Density"]/CRB_Config.video_lane_widths[video]
#     pfd_df["Flow"] = pfd_df["Flow"]/CRB_Config.video_lane_widths[video]
#     pfd_df["Video"] = video
#     pfd_df["Video_Part"] = part
#     if pfd_df_all is None:
#         pfd_df_all = pfd_df.copy()
#     else:
#         pfd_df_all = pd.concat((pfd_df_all, pfd_df))
    
#     del pfd_df
#     counter += 1
#     print(f"... Processed video part {part}. Finished {counter}/{len(CRB_Config.video_parts_X[video])}.")
# gc.collect()

# bin_width = 0.3
# pfd_df_all['Density_Bin'] = pd.cut(pfd_df_all['Density'], bins=np.arange(0, 200.0, bin_width))
# pfd_df_agg = pfd_df_all.groupby(["Density_Bin"], observed=False).agg({
#     "Density": "mean", 
#     "Flow": "mean",
#     "Speed": "mean",
#     "Density_Bin": "count"
# })
# pfd_df_agg = pfd_df_agg.rename(
#     columns={"Density_Bin": "Num_Observations"}
# )
# pfd_df_agg = pfd_df_agg.dropna()
# pfd_df_agg = pfd_df_agg[pfd_df_agg['Num_Observations'] >= 50]

# ts_df_all_Vor['Density'] = ts_df_all_Vor['Density'] * 0.6
# ts_df_all_Vor['Flow'] = ts_df_all_Vor['Flow'] * 0.6
# ts_df_all_Vor['Density_Bin'] = pd.cut(ts_df_all_Vor['Density'], bins=np.arange(0, 200.0, bin_width))
# ts_df_agg = ts_df_all_Vor.groupby(["Density_Bin"], observed=False).agg({
#     "Density": "mean", 
#     "Flow": "mean",
#     "Speed": "mean",
#     "Density_Bin": "count"
# })
# ts_df_agg = ts_df_agg.rename(
#     columns={"Density_Bin": "Num_Observations"}
# )
# ts_df_agg = ts_df_agg.dropna()
# ts_df_agg = ts_df_agg[ts_df_agg['Num_Observations'] >= 50]

# plt.figure()
# plt.scatter(pfd_df_agg['Density'], pfd_df_agg['Flow'])
# plt.scatter(ts_df_agg['Density'], ts_df_agg['Flow'])
# plt.ylim([0, 2500])
# plt.xlim([0, 200])

# plt.figure()
# plt.scatter(pfd_df_agg['Density'], pfd_df_agg['Speed'])
# plt.scatter(ts_df_agg['Density'], ts_df_agg['Speed'])
# plt.ylim([0, 30])
# plt.xlim([0, 200])

# plt.show()

