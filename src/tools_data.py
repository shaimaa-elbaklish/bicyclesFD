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
import numpy as np
import pandas as pd

from dataclasses import dataclass

from _constants import CRB_Config


# #############################################################################
# FUNCTIONS: CRB Dataset
# #############################################################################
def compute_lane_coordinates(df: pd.DataFrame, config: dataclass = CRB_Config) -> pd.DataFrame:
    """
    This function computes the lane cooridnates for a given trajectory dataframe.
    The following columns are added to the dataframe:
        - Polar_X_Int: The continual Polar_X [rad], i.e. does not restart every 2*pi.
        - Lane_X: The radial distance to bicycle [m].
        - Lane_Y_Outer: The longitudinal distance [m], based on outer circle radius.
        - Lane_Y: The longitudinal distance [m], based on actual radial distance of bicycle (i.e. Lane_X).
        - Distance_Travelled: The total distance travelled [m], i.e. includes both lateral and longitunal motions.

    Parameters
    ----------
    df : pd.DataFrame
        The bicycles' trajectories (includes polar coordinates).

    Returns
    -------
    df_mod : pd.DataFrame
        The bicycles' trajectories (includes lane coordinates).

    """
    df_mod = None
    unique_bikes = df["Vehicle_ID"].unique()
    for bike_id in unique_bikes:
        bike_df = df[df["Vehicle_ID"] == bike_id].copy()
        if not bike_df["Frame_ID"].is_monotonic_increasing:
            bike_df = bike_df.sort_values(by="Frame_ID", ascending=True)
        bike_df = bike_df.reset_index().drop(columns="index")
        bike_df["Polar_X_diff"] = bike_df["Polar_X"].diff()
        bike_df.loc[bike_df["Polar_X_diff"]<=-1e-03, "Polar_X_diff"] += 2*np.pi
        bike_df.loc[0, "Polar_X_diff"] = bike_df.loc[0, "Polar_X"]
        bike_df["Polar_X_Int"] = bike_df["Polar_X_diff"].cumsum()    
        bike_df["Lane_X"] = bike_df["Polar_Y"]
        bike_df["Lane_Y_Outer"] = bike_df["Polar_X"]*config.circle_outer_radius
        
        bike_df["Cartesian_X_diff"] = bike_df["Cartesian_X"].diff()
        bike_df["Cartesian_Y_diff"] = bike_df["Cartesian_Y"].diff()
        bike_df["Cartesian_diff"] = np.sqrt(bike_df["Cartesian_X_diff"]**2 + bike_df["Cartesian_Y_diff"]**2)
        bike_df["Distance_Travelled"] = bike_df["Cartesian_diff"].cumsum()

        bike_df["Lane_Y"] = bike_df["Polar_X_diff"] * bike_df["Polar_Y"]
        bike_df["Lane_Y"] = bike_df["Lane_Y"].cumsum()
        
        bike_df = bike_df.drop(columns=["Polar_X_diff", "Cartesian_diff",
                                        "Cartesian_X_diff", "Cartesian_Y_diff"])
        if df_mod is None:
            df_mod = bike_df.copy()
        else:
            df_mod = pd.concat((df_mod, bike_df))
    
    df_mod = df_mod.sort_values(by=["Vehicle_ID", "Frame_ID"], ascending=True)
    df_mod = df_mod.reset_index().drop(columns="index")
    return df_mod


def determine_leader(df, polarX_min_threshold: float = 5*np.pi/180, polarX_max_threshold: float = 60*np.pi/180, 
                     polarY_threshold: float = 0.4, max_lane_width: float = 3.75):
    """
    This function determines the most viable leader for each bicycle.

    Parameters
    ----------
    df : pd.DataFrame
        The bicycles trajectories.
    polarX_min_threshold : float, optional
        The minimum threshold for the visibility region. The default is 5*np.pi/180.
    polarX_max_threshold : float, optional
        The maximum threshold for the visibility region.. The default is 60*np.pi/180.
    polarY_threshold : float, optional
        The lateral coincidence region. The default is 0.4.
    max_lane_width : float, optional
        Maximum lane width set for the CRB experiment. The default is 3.75.

    Returns
    -------
    df : TYPE
        The bicycles trajectories with 'Preceding', 'Time_Hdwy', and 'Space_Hdwy' columns added.

    """
    df[["Preceding", "Space_Hdwy", "Time_Hdwy", "Polar_Y_Dist"]] = pd.NA
    df[["Cartesian_Space_Hdwy", "Cartesian_Time_Hdwy"]] = pd.NA
    grouped = df.groupby(by=['Global_Time'])
    for (t,), group_df in grouped:
        if len(group_df) <= 1:
            continue
        group_df = group_df.sort_values(by='Lane_Y_Outer').reset_index()
        group_df["Polar_X_Quad"] = group_df["Polar_X"].apply(lambda x: x if x <= np.pi else x-2*np.pi)
        for idx, row in group_df.iterrows():
            prec_idx = (idx+1)%len(group_df)
            polarY_diff = abs(group_df.loc[prec_idx, "Polar_Y"] - group_df.loc[idx, "Polar_Y"])
            polarX_diff = abs(group_df.loc[prec_idx, "Polar_X_Quad"] - group_df.loc[idx, "Polar_X_Quad"])
            
            if polarX_diff < polarX_min_threshold:
                prec_idx = (idx+2)%len(group_df)
                polarY_diff = abs(group_df.loc[prec_idx, "Polar_Y"] - group_df.loc[idx, "Polar_Y"])
                polarX_diff = abs(group_df.loc[prec_idx, "Polar_X_Quad"] - group_df.loc[idx, "Polar_X_Quad"])
                        
            if polarY_diff > polarY_threshold:
                polarX_diff2 = abs(group_df.loc[(prec_idx+1)%len(group_df), "Polar_X_Quad"] - group_df.loc[idx, "Polar_X_Quad"])
                polarY_diff2 = abs(group_df.loc[(prec_idx+1)%len(group_df), "Polar_Y"] - group_df.loc[idx, "Polar_Y"])
                if polarY_diff2 <= polarY_threshold and polarX_diff2 <= polarX_max_threshold:
                    prec_idx = (prec_idx+1)%len(group_df)
                    polarX_diff = polarX_diff2
                    polarY_diff = polarY_diff2
                elif polarX_diff2 <= polarX_max_threshold:
                    dist = polarY_diff/max_lane_width + 2*polarX_diff/np.pi
                    dist2 = polarY_diff2/max_lane_width + 2*polarX_diff2/np.pi
                    if dist2 < dist:
                        prec_idx = (prec_idx+1)%len(group_df)
                        polarX_diff = polarX_diff2
                        polarY_diff = polarY_diff2
            
            group_df.loc[idx, "Preceding"] = group_df.loc[prec_idx, "Vehicle_ID"]
            space_hdwy_polar = (group_df.loc[prec_idx, "Polar_X"] - group_df.loc[idx, "Polar_X"])
            if prec_idx < idx:
                space_hdwy_polar = (group_df.loc[prec_idx, "Polar_X"] + 2*np.pi - group_df.loc[idx, "Polar_X"])
            space_hdwy = space_hdwy_polar * 0.5 * (group_df.loc[prec_idx, "Polar_Y"] + group_df.loc[idx, "Polar_Y"])
            cart_space_hdwy = np.sqrt((group_df.loc[prec_idx, "Cartesian_X"] - group_df.loc[idx, "Cartesian_X"])**2 + (group_df.loc[prec_idx, "Cartesian_Y"] - group_df.loc[idx, "Cartesian_Y"])**2)
            time_hdwy = space_hdwy / group_df.loc[idx, "v_Vel"]
            df.loc[row['index'], "Preceding"] = group_df.loc[idx, "Preceding"]
            df.loc[row['index'], "Space_Hdwy"] = space_hdwy
            df.loc[row['index'], "Cartesian_Space_Hdwy"] = cart_space_hdwy
            df.loc[row['index'], "Time_Hdwy"] = time_hdwy
            df.loc[row['index'], "Cartesian_Time_Hdwy"] = cart_space_hdwy / group_df.loc[idx, "v_Vel"]
            df.loc[row['index'], "Polar_Y_Dist"] = abs(group_df.loc[prec_idx, "Polar_Y"] - group_df.loc[idx, "Polar_Y"])
    return df


def determine_leader_Hoogendoorn(df, polarX_min_threshold: float = 5*np.pi/180,
                     polarY_threshold: float = 0.4):
    """
    This function determines the most viable leader for each bicycle.

    Parameters
    ----------
    df : pd.DataFrame
        The bicycles trajectories.
    polarX_min_threshold : float, optional
        The minimum threshold for the visibility region. The default is 5*np.pi/180.
    polarY_threshold : float, optional
        The lateral coincidence region. The default is 0.4.

    Returns
    -------
    df : TYPE
        The bicycles trajectories with 'Preceding', 'Time_Hdwy', and 'Space_Hdwy' columns added.

    """
    df[["Preceding", "Space_Hdwy", "Time_Hdwy", "Polar_Y_Dist"]] = pd.NA
    df[["Cartesian_Space_Hdwy", "Cartesian_Time_Hdwy"]] = pd.NA
    grouped = df.groupby(by=['Global_Time'])
    for (t,), group_df in grouped:
        if len(group_df) <= 1:
            continue
        group_df = group_df.sort_values(by='Lane_Y_Outer', ascending=True).reset_index()
        group_df["Polar_X_Quad"] = group_df["Polar_X"].apply(lambda x: x if x <= np.pi else x-2*np.pi)
        for idx, row in group_df.iterrows():
            possible_leaders = pd.concat((group_df.iloc[idx:], group_df.iloc[:idx]), ignore_index=False)
            possible_leaders = possible_leaders[group_df['Vehicle_ID'] != row['Vehicle_ID']]
            possible_leaders['Polar_Y_Diff'] = (possible_leaders['Polar_Y'] - row['Polar_Y']).abs()
            possible_leaders['Polar_X_Diff'] = possible_leaders['Polar_X'] - row['Polar_X']
            possible_leaders['Polar_X_Diff'] = possible_leaders['Polar_X_Diff'].apply(lambda x: 2*np.pi+x if x <= 0 else x)
            possible_leaders['Condition'] = (possible_leaders['Polar_Y_Diff'] <= polarY_threshold) & \
                (possible_leaders['Polar_X_Diff'] >= polarX_min_threshold)
            prec_idx = possible_leaders['Condition'].idxmax()
            group_df.loc[idx, "Preceding"] = group_df.loc[prec_idx, "Vehicle_ID"]
            space_hdwy_polar = (group_df.loc[prec_idx, "Polar_X"] - group_df.loc[idx, "Polar_X"])
            if prec_idx < idx:
                space_hdwy_polar = (group_df.loc[prec_idx, "Polar_X"] + 2*np.pi - group_df.loc[idx, "Polar_X"])
            space_hdwy = space_hdwy_polar * 0.5 * (group_df.loc[prec_idx, "Polar_Y"] + group_df.loc[idx, "Polar_Y"])
            time_hdwy = space_hdwy / group_df.loc[idx, "v_Vel"]
            cart_space_hdwy = np.sqrt((group_df.loc[prec_idx, "Cartesian_X"] - group_df.loc[idx, "Cartesian_X"])**2 + (group_df.loc[prec_idx, "Cartesian_Y"] - group_df.loc[idx, "Cartesian_Y"])**2)
            df.loc[row['index'], "Preceding"] = group_df.loc[idx, "Preceding"]
            df.loc[row['index'], "Space_Hdwy"] = space_hdwy
            df.loc[row['index'], "Time_Hdwy"] = time_hdwy
            group_df.loc[idx, "Space_Hdwy"] = space_hdwy
            df.loc[row['index'], "Cartesian_Space_Hdwy"] = cart_space_hdwy
            df.loc[row['index'], "Cartesian_Time_Hdwy"] = cart_space_hdwy / group_df.loc[idx, "v_Vel"]
            df.loc[row['index'], "Polar_Y_Dist"] = abs(group_df.loc[prec_idx, "Polar_Y"] - group_df.loc[idx, "Polar_Y"])
            
    return df


def determine_leader_V2(df, polarX_min_threshold: float = 5*np.pi/180, polarX_max_threshold: float = 60*np.pi/180, 
                     polarY_threshold: float = 0.4, max_lane_width: float = 3.75):
    """
    This function determines the most viable leader for each bicycle.

    Parameters
    ----------
    df : pd.DataFrame
        The bicycles trajectories.
    polarX_min_threshold : float, optional
        The minimum threshold for the visibility region. The default is 5*np.pi/180.
    polarX_max_threshold : float, optional
        The maximum threshold for the visibility region.. The default is 60*np.pi/180.
    polarY_threshold : float, optional
        The lateral coincidence region. The default is 0.4.
    max_lane_width : float, optional
        Maximum lane width set for the CRB experiment. The default is 3.75.

    Returns
    -------
    df : TYPE
        The bicycles trajectories with 'Preceding', 'Time_Hdwy', and 'Space_Hdwy' columns added.

    """
    df[["Preceding", "Space_Hdwy", "Time_Hdwy", "Polar_Y_Dist"]] = pd.NA
    df[["Cartesian_Space_Hdwy", "Cartesian_Time_Hdwy"]] = pd.NA
    grouped = df.groupby(by=['Global_Time'])
    for (t,), group_df in grouped:
        if len(group_df) <= 1:
            continue
        group_df = group_df.sort_values(by='Lane_Y_Outer', ascending=True).reset_index()
        group_df["Polar_X_Quad"] = group_df["Polar_X"].apply(lambda x: x if x <= np.pi else x-2*np.pi)
        for idx, row in group_df.iterrows():
            possible_leaders = pd.concat((group_df.iloc[idx:], group_df.iloc[:idx]), ignore_index=False)
            possible_leaders = possible_leaders[group_df['Vehicle_ID'] != row['Vehicle_ID']]
            possible_leaders['Polar_Y_Diff'] = (possible_leaders['Polar_Y'] - row['Polar_Y']).abs()
            possible_leaders['Polar_X_Diff'] = possible_leaders['Polar_X'] - row['Polar_X']
            possible_leaders['Polar_X_Diff'] = possible_leaders['Polar_X_Diff'].apply(lambda x: 2*np.pi+x if x <= 0 else x)
            # Problem (1)
            possible_leaders['Condition'] = (possible_leaders['Polar_Y_Diff'] <= polarY_threshold) & \
                (possible_leaders['Polar_X_Diff'] >= polarX_min_threshold)
            prec_idx = possible_leaders['Condition'].idxmax()
            if possible_leaders.loc[prec_idx, 'Polar_X_Diff'] > polarX_max_threshold:
                # Go to problem (2)
                possible_leaders['Combined_Dist'] = possible_leaders['Polar_Y_Diff']/max_lane_width + \
                                                    2*possible_leaders['Polar_X_Diff']/np.pi + \
                                                    1000*(possible_leaders['Polar_X_Diff'] <= polarX_min_threshold)
                prec_idx = possible_leaders['Combined_Dist'].idxmin()
                # possible_leaders['Polar_X'] = possible_leaders['Polar_X'] * 180 / np.pi
                # possible_leaders['Polar_X_Quad'] = possible_leaders['Polar_X_Quad'] * 180 / np.pi
                # possible_leaders['Polar_X_Diff'] = possible_leaders['Polar_X_Diff'] * 180 / np.pi
                # print(possible_leaders[['Vehicle_ID', 'Polar_X', 'Polar_X_Quad', 'Polar_X_Diff', 'Polar_Y_Diff']])
                # print(row['Vehicle_ID'], row['Polar_X']*180/np.pi, prec_idx)
                # print(possible_leaders[['Vehicle_ID', 'Condition', 'Combined_Dist']])
                # sys.exit(1)
                
            group_df.loc[idx, "Preceding"] = group_df.loc[prec_idx, "Vehicle_ID"]
            space_hdwy_polar = (group_df.loc[prec_idx, "Polar_X"] - group_df.loc[idx, "Polar_X"])
            if prec_idx < idx:
                space_hdwy_polar = (group_df.loc[prec_idx, "Polar_X"] + 2*np.pi - group_df.loc[idx, "Polar_X"])
            space_hdwy = space_hdwy_polar * 0.5 * (group_df.loc[prec_idx, "Polar_Y"] + group_df.loc[idx, "Polar_Y"])
            time_hdwy = space_hdwy / group_df.loc[idx, "v_Vel"]
            cart_space_hdwy = np.sqrt((group_df.loc[prec_idx, "Cartesian_X"] - group_df.loc[idx, "Cartesian_X"])**2 + (group_df.loc[prec_idx, "Cartesian_Y"] - group_df.loc[idx, "Cartesian_Y"])**2)
            df.loc[row['index'], "Preceding"] = group_df.loc[idx, "Preceding"]
            df.loc[row['index'], "Space_Hdwy"] = space_hdwy
            df.loc[row['index'], "Time_Hdwy"] = time_hdwy
            group_df.loc[idx, "Space_Hdwy"] = space_hdwy
            df.loc[row['index'], "Cartesian_Space_Hdwy"] = cart_space_hdwy
            df.loc[row['index'], "Cartesian_Time_Hdwy"] = cart_space_hdwy / group_df.loc[idx, "v_Vel"]
            df.loc[row['index'], "Polar_Y_Dist"] = abs(group_df.loc[prec_idx, "Polar_Y"] - group_df.loc[idx, "Polar_Y"])
            
    return df