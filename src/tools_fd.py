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
import sys
import logging
import numpy as np
import pandas as pd

from dataclasses import dataclass

from _constants import CRB_Config


# #############################################################################
# FUNCTIONS
# #############################################################################
def compute_pseudo_states_ssd(df: pd.DataFrame, num_sectors: int = 5, 
                              dt: float = 4.0, lane_width: float = 1.0,
                              config: dataclass = CRB_Config, 
                              TTD_USE_POLAR: bool = False) -> pd.DataFrame:
    """
    This function computes the pseudo-traffic states using a static spatial discretization;
    where the circular track is split into sectors and Eddie's generalized defintions are applied.

    Parameters
    ----------
    df : pd.DataFrame
        The bicycles' trajectories (includes lane coordinates).
    num_sectors : int, optional
        The number of sectors into which the circular track is split. The default is 5.
    dt : float, optional
        The temporal discretization in seconds. The default is 4.0.
    lane_width : float, optional
        The radial cut [m], to define the lane width in which the bicycles are in. The default is 1.0.

    Returns
    -------
    ts_df : pd.DataFrame
        The pseudo-traffic states computed.

    """
    subdf = df.copy()
    subdf = subdf.sort_values(by=['Vehicle_ID', 'Frame_ID'], ascending=True)
    subdf = subdf.reset_index().drop(columns='index')
    subdf = subdf[subdf["Lane_X"] >= CRB_Config.circle_outer_radius - lane_width]
    subdf['Time_Bin'] = pd.cut(x=subdf['Global_Time'], bins=np.arange(0, df['Global_Time'].max(), dt))
    subdf['Polar_X_Bin'] = pd.cut(x=subdf['Polar_X'], bins=np.linspace(0, 2*np.pi, num_sectors+1))
    grouped = subdf.groupby(by=["Time_Bin", "Polar_X_Bin", "Vehicle_ID"], observed=False)
    ttt_res, ttd_res, vehID_res = {}, {}, {}
    for (time_bin, polarX_bin, veh_id), group_df in grouped:
        if len(group_df) <= 1: 
            continue
        if time_bin not in ttt_res.keys():
            ttt_res[time_bin], ttd_res[time_bin] = {}, {}
            vehID_res[time_bin] = {}
            ttt_res[time_bin][polarX_bin], ttd_res[time_bin][polarX_bin] = 0, 0
            vehID_res[time_bin][polarX_bin] = []
        elif polarX_bin not in ttt_res[time_bin].keys():
            ttt_res[time_bin][polarX_bin], ttd_res[time_bin][polarX_bin] = 0, 0
            vehID_res[time_bin][polarX_bin] = []
        ttt_res[time_bin][polarX_bin] += (group_df["Global_Time"].max() - group_df["Global_Time"].min())
        if TTD_USE_POLAR:
            ttd_res[time_bin][polarX_bin] += (group_df["Polar_X_Int"].max() - group_df["Polar_X_Int"].min())*group_df["Polar_Y"].mean()
        else:
            ttd_res[time_bin][polarX_bin] += (group_df["Distance_Travelled"].max() - group_df["Distance_Travelled"].min())
        vehID_res[time_bin][polarX_bin].append(veh_id)
    ts_df = {
        "Global_Time_Bin": [], "Polar_X_Bin": [], "Vehicle_IDs": [], 
        "TTT": [], "TTD": []
    }
    for time_bin in ttt_res.keys():
        for polarX_bin in ttt_res[time_bin].keys():
            ts_df["Global_Time_Bin"].append(time_bin)
            ts_df["Polar_X_Bin"].append(polarX_bin)
            ts_df["Vehicle_IDs"].append(vehID_res[time_bin][polarX_bin])
            ts_df["TTT"].append(ttt_res[time_bin][polarX_bin])
            ts_df["TTD"].append(ttd_res[time_bin][polarX_bin])
    ts_df = pd.DataFrame(ts_df)
    ts_df["Num_Bicycles"] = ts_df["Vehicle_IDs"].apply(lambda x: len(x))
    ts_df["Speed"] = (ts_df["TTD"]/1000.0) / (ts_df["TTT"]/3600.0)
    area = (2*np.pi/num_sectors)*(CRB_Config.circle_outer_radius-0.5*lane_width)/1000.0 * dt/3600.0
    ts_df["Density"] = (ts_df["TTT"]/3600.0) / area
    ts_df["Flow"] = (ts_df["TTD"]/1000.0) / area
    return ts_df