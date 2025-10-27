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
import ast

import numpy as np
import pandas as pd
import lmfit as lm
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple

from _log_config import logger
from _constants import CRB_Config


# #############################################################################
# FUNCTIONS
# #############################################################################
def compute_pseudo_states_pfd_N2(df: pd.DataFrame, lane_width: float, 
                                 config: dataclass = CRB_Config) -> pd.DataFrame:
    """
    This function computes the pseudo-traffic states based on the BFD method.

    Parameters
    ----------
    df : pd.DataFrame
        Bicycles trajectories.
    lane_width : float
        Lane width setting in meters.
    config : dataclass, optional
        Dataset configuration. The default is CRB_Config.

    Returns
    -------
    grouped : TYPE
        Pseudo-traffic states dataframe.
    """
    if 'Space_Hdwy' not in df.columns:
        print("ERROR: Please make sure you leader-follower pairs are identified at first and Space_Hdwy is computed.")
        sys.exit(1)
        return
    subdf = df.copy()
    # subdf = determine_leader(subdf)
    subdf = subdf.sort_values(by=['Vehicle_ID', 'Frame_ID'], ascending=True)
    subdf = subdf.reset_index().drop(columns='index')
    subdf[['Next_Polar_X_Int', 'Next_Polar_Y', 'Next_Lane_Y']] = subdf.groupby('Vehicle_ID')[['Polar_X_Int', 'Polar_Y', 'Lane_Y']].shift(-1)
    subdf[['Next_Space_Hdwy', 'Next_Cartesian_Space_Hdwy', 'Next_Polar_Y_Dist']] = subdf.groupby('Vehicle_ID')[['Space_Hdwy', 'Cartesian_Space_Hdwy', 'Polar_Y_Dist']].shift(-1)
    subdf = subdf.drop(subdf[subdf["Frame_ID"]==subdf["Frame_ID"].max()].index)
    subdf = subdf.dropna()
    subdf = subdf[subdf["Lane_X"] >= config.circle_outer_radius - lane_width]
    
    FPS = config.sampling_freq
    num_vehicles = 2
    avg_bike_length = 1.8 # m 
    grouped = subdf[["Global_Time", "Vehicle_ID", "Preceding", "Polar_X", "Time_Hdwy",
                     "Polar_X_Int", "Next_Polar_X_Int", "Polar_Y", "Next_Polar_Y", 
                     "Space_Hdwy", "Next_Space_Hdwy", "Lane_Y", "Next_Lane_Y",
                     "Cartesian_Space_Hdwy", "Next_Cartesian_Space_Hdwy",
                     "Polar_Y_Dist", "Next_Polar_Y_Dist"]].copy()
    grouped["TTT"] =  (1/FPS) * (num_vehicles-1) / 3600.0 # hour
    # grouped["x0"] = grouped["Lane_Y"]
    # grouped["xt"] = grouped["Next_Lane_Y"]
    grouped["x0"] = grouped["Polar_X_Int"] * 0.5*(grouped["Polar_Y"] + grouped["Next_Polar_Y"])
    grouped["xt"] = grouped["Next_Polar_X_Int"] * 0.5*(grouped["Polar_Y"] + grouped["Next_Polar_Y"])
    
    grouped["xL0"] = grouped["x0"] + grouped["Space_Hdwy"]
    grouped["xLt"] = grouped["xt"] + grouped["Next_Space_Hdwy"]
    
    grouped["TTD"] = abs(grouped["xt"]-grouped["x0"]) / 1000.0 # km
    grouped["Area"] = 0.5*(1/FPS/3600.0)*(grouped["xL0"]-grouped["x0"] + grouped["xLt"]-grouped["xt"] + 2*avg_bike_length)/1000.0 # km.h
    grouped["Area"] = grouped["Area"].astype(np.float64).clip(lower=0)
    grouped = grouped[grouped['Area'] > 0]
    grouped["Density"] = grouped["TTT"] / grouped["Area"]
    grouped["Flow"] = grouped["TTD"] / grouped["Area"]
    grouped["Speed"] = grouped["Flow"] / grouped["Density"]
    grouped["Vehicle_IDs"] = grouped[["Preceding", "Vehicle_ID"]].values.tolist()
    
    grouped = grouped[["Global_Time", "Vehicle_IDs", "Density", "Flow", "Speed", 
                       "Area", "TTT", "TTD", "Polar_Y", "Polar_X", "Time_Hdwy", 
                       "Space_Hdwy", "Cartesian_Space_Hdwy", "Polar_Y_Dist"]]
    
    grouped = grouped.dropna()
    return grouped


def aggregate_fd(ts_df: pd.DataFrame, max_density: float, bin_width: float, 
                  min_observations: int = 10, FD_form: str = "ExpFD", 
                  loss_fn: str = "NRMSE", jam_density: float = None, 
                  show_pseudo_states: bool = True, log_results: bool = False):
    """
    This function aggregates the pseudo-traffic states within a given density
    bin width and calibrates a functional form to the aggregated states. 
    The resulting FDs (k-q and k-v) are plotted.

    Parameters
    ----------
    ts_df : pd.DataFrame
        The pseudo-traffic states dataframe.
    max_density : float
        Maximum density in bic/km to generate the density bins.
    bin_width : float
        The density bin width in bic/km to generate the density bins.
    min_observations : int, optional
        The minimum number of observation in a single bin for aggregation. The default is 10.
    FD_form : str, optional
        The functional form of the FD to calibrate. The default is "ExpFD".
    loss_fn : str, optional
        The loss function to calibrate the FD functional form, either "NRMSE" or "HuberLoss". The default is "NRMSE".
    jam_density : float
        Estimate of the jam density in bic/km to generate better congested FD shape. 
        If None, estimate jam density through FD calibration. The default is None.
    show_pseudo_states: bool, optional
        If true, show the pseudo-traffic states in the poltted FD diagram. The default is True.
    log_results: bool, optional
        If true, log results to global log file already open through _log_config.py. The default is False.

    Returns
    -------
    agg_df : pd.DataFrame
        The aggregated traffic states dataframe.
    fig: plt.Figure
        The constructed FD (k-q and k-v diagrams).
    """
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
    if log_results:
        logger.info(
            f"""Num_Observations Statistics: 
                Min = {agg_df['Num_Observations'].min()}, Max = {agg_df['Num_Observations'].max()}, 
                Mean = {agg_df['Num_Observations'].mean():.1f}, Median = {agg_df['Num_Observations'].median()}"""
        )
    else:
        print(agg_df["Num_Observations"].min(), agg_df["Num_Observations"].max())
        print(agg_df["Num_Observations"].mean(), agg_df["Num_Observations"].median())

    # add check for number of observations
    if agg_df["Num_Observations"].max() <= min_observations:
        if log_results:
            logger.warning(f"Too low observations for min_observations = {min_observations}. Reset min_observations = {int(agg_df['Num_Observations'].mean())}.")
        else:
            print(f"Reset min_observations from {min_observations} to {int(agg_df['Num_Observations'].mean())}.")
        min_observations = int(agg_df["Num_Observations"].mean())
    
    agg_df = agg_df[agg_df["Num_Observations"] >= min_observations]
        
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    if show_pseudo_states:
        axs[0].scatter(ts_df["Density"], ts_df["Flow"], label="Pseudo-States", alpha=0.1)
    axs[0].scatter(agg_df["Density"], agg_df["Flow"], label="Aggregated")
    axs[0].set_xlabel("Density [bic/km/m]")
    axs[0].set_ylabel("Flow [bic/h/m]")
    axs[0].set_ylim([0, 2000])
    axs[0].set_xlim([0, 200])
    if show_pseudo_states:
        axs[1].scatter(ts_df["Density"], ts_df["Speed"], label="Pseudo-States", alpha=0.1)
    axs[1].scatter(agg_df["Density"], agg_df["Speed"], label="Aggregated")
    axs[1].set_xlabel("Density [bic/km/m]")
    axs[1].set_ylabel("Speed [km/h]")
    axs[1].set_ylim([0, 20])
    axs[1].set_xlim([0, 200])
    
    if FD_form == 'ExpFD':
        res = _calibrate_FD(
            Ks=agg_df["Density"].to_numpy().astype(np.float32), 
            Qs=agg_df["Flow"].to_numpy().astype(np.float32), 
            Vs=agg_df["Speed"].to_numpy().astype(np.float32),
            FD_form=FD_form, loss_fn=loss_fn, log_results=log_results
        )
        k_FD, q_FD, v_FD = res[0], res[1], res[2]
        axs[0].plot(k_FD, q_FD, label="k-q FD", linestyle="dashed", color="black")
        axs[1].plot(k_FD, v_FD, label="k-v FD", linestyle="dashed", color="black")
    elif FD_form == 'WuFD':
        Ks = agg_df["Density"].to_numpy().astype(np.float32)
        Qs = agg_df["Flow"].to_numpy().astype(np.float32)
        Vs = agg_df["Speed"].to_numpy().astype(np.float32)
        res = _calibrate_FD(Ks=Ks, Qs=Qs, Vs=Vs, FD_form='WuFreeFD', loss_fn=loss_fn, log_results=log_results)
        k_FD, q_FD, v_FD = res[0], res[1], res[2]
        # WuFreeFD returns: K_test, Q_pred_free, V_pred_free, vf, v_crit, delta, k_crit
        k_crit = res[-1]
        k_FD, q_FD, v_FD = res[0], res[1], res[2]
        Q_cap = np.amax(q_FD)
        axs[0].plot(k_FD[k_FD <= k_crit*1.05], q_FD[k_FD <= k_crit*1.05], label="k-q Free-Flow FD", linestyle="dashed", color="blue")
        axs[1].plot(k_FD[k_FD <= k_crit*1.05], v_FD[k_FD <= k_crit*1.05], label="k-v Free-Flow FD", linestyle="dashed", color="blue")
        
        if jam_density is None:
            jam_density = k_crit * np.power(res[-4]/(res[-4]-res[-3]), 1/res[-2]) # np.power(vf/(vf-v_cr), 1/delta) * k_crit
        cong_idxs = (Ks >= k_crit*0.5) & (Qs <= Q_cap)
        res = _calibrate_FD(Ks=Ks[cong_idxs], Qs=Qs[cong_idxs], Vs=Vs[cong_idxs], FD_form='WuCongFD', loss_fn=loss_fn, log_results=log_results, k_jam_est=jam_density)
        # WuCongFD returns: K_test, Q_pred_cong, V_pred_cong, k_jam, w
        k_FD, q_FD, v_FD = res[0], res[1], res[2]
        axs[0].plot(k_FD[k_FD >= k_crit*0.5], q_FD[k_FD >= k_crit*0.5], label="k-q Congested FD", linestyle="dashed", color="red")
        axs[1].plot(k_FD[k_FD >= k_crit*0.5], v_FD[k_FD >= k_crit*0.5], label="k-v Congested FD", linestyle="dashed", color="red")
    
    # axs[0].legend(loc='upper right')
    # axs[1].legend(loc='upper right')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, bbox_to_anchor=(0.5, -0.05), loc='lower center', ncol=3, bbox_transform=fig.transFigure)
    fig.tight_layout()
    
    return agg_df, fig


def _expFD(Ks, vf, alpha, k_crit):
    V_pred = vf * np.exp(-np.power(Ks/k_crit, alpha)/alpha)
    Q_pred = Ks * V_pred
    return V_pred, Q_pred


def _WuFreeFD(Ks, vf, v_crit, delta, k_crit):
    V_pred_free = np.maximum(0, vf - (vf-v_crit)*np.power(Ks/k_crit, delta))
    Q_pred_free = Ks * V_pred_free
    return V_pred_free, Q_pred_free


def _WuCongFD(Ks, w, k_jam):
    Q_pred_cong = np.maximum(0, w*(Ks - k_jam))
    V_pred_cong = Q_pred_cong / Ks
    return V_pred_cong, Q_pred_cong


def _nrmse(params, Ks, Qs, Vs, FD_form):
    if FD_form == "ExpFD":
        vf, alpha = params['vf'], params['alpha']
        k_crit = params['k_crit']
        V_pred, Q_pred = _expFD(Ks, vf, alpha, k_crit)
    elif FD_form == "WuFreeFD":
        vf, delta = params['vf'], params['delta']
        k_crit, v_crit = params['k_crit'], params['v_crit']
        V_pred, Q_pred = _WuFreeFD(Ks, vf, v_crit, delta, k_crit,)
    elif FD_form == "WuCongFD":
        k_jam, w = params['k_jam'], params['w']
        V_pred, Q_pred = _WuCongFD(Ks, w, k_jam)
    else:
        raise NotImplementedError()
    rmse_Q = np.sqrt(np.mean(np.square(Q_pred - Qs)))
    rmse_V = np.sqrt(np.mean(np.square(V_pred - Vs)))
    obj = rmse_Q/np.mean(Qs) + rmse_V/np.mean(Vs)
    return obj


def _huberLoss(params, Ks, Qs, Vs, FD_form, deltaH_Q=10.0, deltaH_V=1.0):
    if FD_form == "ExpFD":
        vf, alpha = params['vf'], params['alpha']
        k_crit = params['k_crit']
        V_pred, Q_pred = _expFD(Ks, vf, alpha, k_crit)
    elif FD_form == "WuFreeFD":
        vf, delta = params['vf'], params['delta']
        k_crit, v_crit = params['k_crit'], params['v_crit']
        V_pred, Q_pred = _WuFreeFD(Ks, vf, v_crit, delta, k_crit)
    elif FD_form == 'WuCongFD':
        k_jam, w = params['k_jam'], params['w']
        V_pred, Q_pred = _WuCongFD(Ks, w, k_jam)
    else:
        raise NotImplementedError()
    abs_diff_Q = np.abs(Q_pred - Qs)
    loss_Q = deltaH_Q * (abs_diff_Q - 0.5*deltaH_Q)
    loss_Q[abs_diff_Q <= deltaH_Q] = 0.5*np.square(abs_diff_Q[abs_diff_Q <= deltaH_Q])
    
    abs_diff_V = np.abs(V_pred - Vs)
    loss_V = deltaH_V * (abs_diff_V - 0.5*deltaH_V)
    loss_V[abs_diff_V <= deltaH_V] = 0.5*np.square(abs_diff_V[abs_diff_V <= deltaH_V])
    
    obj = np.sum(loss_Q)/np.mean(Qs) + np.sum(loss_V)/np.mean(Vs)
    return obj


def _calibrate_FD(Ks, Qs, Vs, FD_form = "ExpFD", loss_fn = "NRMSE", k_jam_est=None, log_results=False):
    if FD_form == "ExpFD":
        params = lm.create_params(
            vf = {'value': 2, 'min': 1e-05, 'max': 20},
            alpha = {'value': 5, 'min': 1e-05, 'max': 50},
            k_crit = {'value': 100, 'min': 1e-05, 'max': 120}
        )
    elif FD_form == "WuFreeFD":
        params = lm.create_params(
            vf = {'value': 2, 'min': 1e-05, 'max': 20},
            delta = {'value': 0.5, 'min': 1e-05, 'max': 10},
            k_crit = {'value': 100, 'min': 50.0, 'max': 120},
            v_crit = {'value': 2, 'min': 1e-05, 'max': 20},
        )
    elif FD_form == "WuCongFD":
        if k_jam_est is None:
            params = lm.create_params(
                k_jam = {'value': 120, 'min': 50, 'max': 200},
                w = {'value': -2, 'min': -40, 'max': -1e-05},
            )
        else:
            params = lm.create_params(
                k_jam = {'value': k_jam_est, 'vary': False},
                w = {'value': -2, 'min': -40, 'max': -1e-05},
            )
    else:
        raise NotImplementedError()
    if loss_fn == "NRMSE":
        res = lm.minimize(_nrmse, params, args=(Ks, Qs, Vs, FD_form), method='differential_evolution')
    elif loss_fn == "HuberLoss":
        res = lm.minimize(_huberLoss, params, args=(Ks, Qs, Vs, FD_form), method='differential_evolution')
    else:
        raise NotImplementedError()
    if log_results:
        logger.info(
            f"""FD calibration. 
            Loss function: {loss_fn}. FD functional form: {FD_form}.
            Results: {lm.fit_report(res.params)}""")
    else:
        print(lm.fit_report(res.params))
    if FD_form == "ExpFD":
        vf, alpha = res.params['vf'].value, res.params['alpha'].value
        k_crit = res.params['k_crit'].value
        K_test = np.linspace(0, 2*k_crit, 200)
        V_pred, Q_pred = _expFD(K_test, vf, alpha, k_crit)
        return K_test, Q_pred, V_pred, vf, alpha, k_crit
    elif FD_form == "WuFreeFD":
        vf, delta = res.params['vf'].value, res.params['delta'].value
        k_crit, v_crit = res.params['k_crit'].value, res.params['v_crit'].value
        K_test = np.linspace(0, 2*k_crit, 200)
        V_pred_free, Q_pred_free = _WuFreeFD(K_test, vf, v_crit, delta, k_crit)
        return K_test, Q_pred_free, V_pred_free, vf, v_crit, delta, k_crit
    elif FD_form == "WuCongFD":
        k_jam, w = res.params['k_jam'].value, res.params['w'].value
        K_test = np.linspace(0, k_jam, 200)
        V_pred_cong, Q_pred_cong = _WuCongFD(K_test, w, k_jam)
        return K_test, Q_pred_cong, V_pred_cong, k_jam, w
    else:
        raise NotImplementedError()


def estimate_traffic_states(orig_pfd_df: pd.DataFrame, trajectory_df: pd.DataFrame,
                            dx: float, dy: float, dt: float, lane_width: float, 
                            config: dataclass = CRB_Config, mode: str = "XY") -> Tuple[np.ndarray]:
    pfd_df = orig_pfd_df.copy()
    pfd_df['Vehicle_IDs'] = pfd_df['Vehicle_IDs'].apply(lambda x: ast.literal_eval(x))
    pfd_df['Preceding_ID'] = pfd_df['Vehicle_IDs'].apply(lambda x: x[0])
    pfd_df['Ego_ID'] = pfd_df['Vehicle_IDs'].apply(lambda x: x[1])

    trajectory_df = trajectory_df.rename(
        columns={'Vehicle_ID': 'Ego_ID', 'Polar_X': 'Ego_Polar_X', 'Polar_Y': 'Ego_Polar_Y'}
    )
    pfd_df = pfd_df.merge(trajectory_df[['Global_Time', 'Ego_ID', 'Ego_Polar_X', 'Ego_Polar_Y']],
                          on=['Global_Time', 'Ego_ID'], how='left')
    trajectory_df = trajectory_df.rename(
        columns={'Ego_ID': 'Preceding_ID', 'Ego_Polar_X': 'Preceding_Polar_X', 'Ego_Polar_Y': 'Preceding_Polar_Y'}
    )
    pfd_df = pfd_df.merge(trajectory_df[['Global_Time', 'Preceding_ID', 'Preceding_Polar_X', 'Preceding_Polar_Y']],
                          on=['Global_Time', 'Preceding_ID'], how='left')
    
    pfd_df = pfd_df[['Global_Time', 'Ego_ID', 'Preceding_ID', 'Density', 'Flow', 'Speed',
                     'Ego_Polar_X', 'Ego_Polar_Y', 'Preceding_Polar_X', 'Preceding_Polar_Y']]

    time_bins = np.arange(pfd_df['Global_Time'].min(), pfd_df['Global_Time'].max()+dt, dt)
    time_bins = np.round(time_bins, decimals=6)
    polar_x_bins = np.arange(0, 2*np.pi+dx, dx)
    tol = 4*dy
    polar_y_bins = np.arange(config.circle_outer_radius-lane_width-tol, config.circle_outer_radius+dy+tol, dy)
    bike_size = (1.8 + 0.5, 0.8 + 0.2)
    pfd_df['Ego_Polar_X'] = pfd_df['Ego_Polar_X'] - 0.5*bike_size[0]/pfd_df['Ego_Polar_Y']
    pfd_df.loc[pfd_df['Ego_Polar_X'] < 0, 'Ego_Polar_X'] += 2*np.pi
    pfd_df['Preceding_Polar_X'] = pfd_df['Preceding_Polar_X'] + 0.5*bike_size[0]/pfd_df['Preceding_Polar_Y']
    pfd_df.loc[pfd_df['Preceding_Polar_X'] > 2*np.pi, 'Preceding_Polar_X'] -= 2*np.pi
    pfd_df['Ego_Polar_Y'] = pfd_df.apply(lambda row: row['Ego_Polar_Y']+0.5*bike_size[1] if row['Ego_Polar_Y']>row['Preceding_Polar_Y'] else row['Ego_Polar_Y']-0.5*bike_size[1], axis=1)
    pfd_df['Preceding_Polar_Y'] = pfd_df.apply(lambda row: row['Preceding_Polar_Y']+0.5*bike_size[1] if row['Ego_Polar_Y']<row['Preceding_Polar_Y'] else row['Preceding_Polar_Y']-0.5*bike_size[1], axis=1)
    
    # Binning
    pfd_df['Time_Bin_Num'] = np.digitize(pfd_df['Global_Time'].to_numpy(), time_bins, right=True)
    pfd_df['Ego_Polar_X_Bin_Num'] = np.digitize(pfd_df['Ego_Polar_X'].to_numpy(), polar_x_bins, right=True)
    pfd_df['Ego_Polar_Y_Bin_Num'] = np.digitize(pfd_df['Ego_Polar_Y'].to_numpy(), polar_y_bins, right=True)
    pfd_df['Preceding_Polar_X_Bin_Num'] = np.digitize(pfd_df['Preceding_Polar_X'].to_numpy(), polar_x_bins, right=True)
    pfd_df['Preceding_Polar_Y_Bin_Num'] = np.digitize(pfd_df['Preceding_Polar_Y'].to_numpy(), polar_y_bins, right=True)

    pfd_df.loc[pfd_df['Global_Time'] == pfd_df['Global_Time'].min(), 'Time_Bin_Num'] = 1
    pfd_df['Time_Bin_Num'] = pfd_df['Time_Bin_Num'].apply(lambda x: np.nan if x == 0 or x == len(time_bins) else x)
    pfd_df['Ego_Polar_X_Bin_Num'] = pfd_df['Ego_Polar_X_Bin_Num'].apply(lambda x: np.nan if x == 0 or x == len(polar_x_bins) else x)
    pfd_df['Preceding_Polar_X_Bin_Num'] = pfd_df['Preceding_Polar_X_Bin_Num'].apply(lambda x: np.nan if x == 0 or x == len(polar_x_bins) else x)
    pfd_df['Ego_Polar_Y_Bin_Num'] = pfd_df['Ego_Polar_Y_Bin_Num'].apply(lambda x: np.nan if x == 0 or x == len(polar_y_bins) else x)
    pfd_df['Preceding_Polar_Y_Bin_Num'] = pfd_df['Preceding_Polar_Y_Bin_Num'].apply(lambda x: np.nan if x == 0 or x == len(polar_y_bins) else x)
    pfd_df = pfd_df.dropna()
    pfd_df = pfd_df.astype({
        'Time_Bin_Num': 'int', 'Ego_Polar_X_Bin_Num': 'int', 'Preceding_Polar_X_Bin_Num': 'int',
        'Ego_Polar_Y_Bin_Num': 'int', 'Preceding_Polar_Y_Bin_Num': 'int'
    })
    
    # Total Area
    pfd_df['Tot_Dist_Polar_X'] = pfd_df['Preceding_Polar_X'] - pfd_df['Ego_Polar_X']
    pfd_df.loc[pfd_df['Tot_Dist_Polar_X'] < 0, 'Tot_Dist_Polar_X'] += 2*np.pi
    pfd_df['Tot_Dist_Polar_Y'] = abs(pfd_df['Preceding_Polar_Y'] - pfd_df['Ego_Polar_Y'])

    # Get weights for contributions
    pfd_df['Polar_X_Idxs'] = pfd_df[['Ego_Polar_X_Bin_Num', 'Preceding_Polar_X_Bin_Num']].values.tolist()
    pfd_df['Polar_X_Idxs'] = pfd_df['Polar_X_Idxs'].apply(lambda x: [(x[0]-1+i) % (len(polar_x_bins)-1) for i in range((x[1]-x[0])%(len(polar_x_bins)-1) + 1)])
    pfd_df['Polar_Y_Idxs'] = pfd_df[['Ego_Polar_Y_Bin_Num', 'Preceding_Polar_Y_Bin_Num']].values.tolist()
    pfd_df['Polar_Y_Idxs'] = pfd_df['Polar_Y_Idxs'].apply(lambda x: list(range(min(x[0], x[1])-1, max(x[0], x[1]))))

    pfd_df['W_X_Start'] = pfd_df.apply(lambda row: polar_x_bins[row['Polar_X_Idxs'][0]+1] - row['Ego_Polar_X'] , axis=1)
    pfd_df['W_X_End'] = pfd_df.apply(lambda row: row['Preceding_Polar_X'] - polar_x_bins[row['Polar_X_Idxs'][-1]] , axis=1)
    pfd_df['W_X'] = pfd_df.apply(lambda row: [row['W_X_Start']] + [dx]*(len(row['Polar_X_Idxs'])-2) + [row['W_X_End']] if len(row['Polar_X_Idxs']) >= 2 else [row['Tot_Dist_Polar_X']], axis=1)
    pfd_df = pfd_df.drop(columns=['W_X_Start', 'W_X_End'])


    pfd_df['W_Y_Start'] = pfd_df.apply(lambda row: polar_y_bins[row['Polar_Y_Idxs'][0]+1] - min(row['Ego_Polar_Y'], row['Preceding_Polar_Y']) , axis=1)
    pfd_df['W_Y_End'] = pfd_df.apply(lambda row: max(row['Ego_Polar_Y'], row['Preceding_Polar_Y']) - polar_y_bins[row['Polar_Y_Idxs'][-1]] , axis=1)
    pfd_df['W_Y'] = pfd_df.apply(lambda row: [row['W_Y_Start']] + [dy]*(len(row['Polar_Y_Idxs'])-2) + [row['W_Y_End']] if len(row['Polar_Y_Idxs']) >= 2 else [row['Tot_Dist_Polar_Y']], axis=1)
    pfd_df = pfd_df.drop(columns=['W_Y_Start', 'W_Y_End'])

    def fill_weights(row):
        wx, wy = np.asarray(row['W_X'])/row['Tot_Dist_Polar_X'], np.asarray(row['W_Y'])/row['Tot_Dist_Polar_Y']
        x_idxs, y_idxs = np.asarray(row['Polar_X_Idxs']), np.asarray(row['Polar_Y_Idxs'])
        if mode == "XY":
            XY = np.zeros((len(polar_x_bins)-1, len(polar_y_bins)-1))
            XY[np.ix_(x_idxs, y_idxs)] = np.outer(wx, wy)
            return XY
        if mode == "X":
            X = np.zeros((len(polar_x_bins)-1,))
            X[x_idxs] = wx
            return X
        if mode == "Y":
            Y = np.zeros((len(polar_y_bins)-1,))
            Y[y_idxs] = wy
            return Y

    pfd_df['W_XY'] = pfd_df.apply(fill_weights, axis=1)
    pfd_df['Density_W_XY'] = pfd_df['Density'] * pfd_df['W_XY']
    pfd_df['Flow_W_XY'] = pfd_df['Flow'] * pfd_df['W_XY']
    pfd_df['Speed_W_XY'] = pfd_df['Speed'] * pfd_df['W_XY']
    pfd_df['Num_Obs_W_XY'] = pfd_df['W_XY'].apply(lambda x: (100*x != 0).astype(int))
    grouped = pfd_df.groupby(by='Time_Bin_Num', observed=False).agg(
        Num_Observations=pd.NamedAgg(column="Num_Obs_W_XY", aggfunc="sum"),
        Density=pd.NamedAgg(column="Density_W_XY", aggfunc="sum"),
        Flow=pd.NamedAgg(column="Flow_W_XY", aggfunc="sum"),
        Speed=pd.NamedAgg(column="Speed_W_XY", aggfunc="sum"),
    )
    grouped = grouped.reset_index().dropna()
    grouped = grouped.sort_values(by='Time_Bin_Num', ascending=True)
    assert(len(grouped) == len(time_bins)-1)
    Density_Mat = np.stack(grouped['Density'].to_numpy())
    Flow_Mat = np.stack(grouped['Flow'].to_numpy())
    Speed_Mat = np.stack(grouped['Speed'].to_numpy())
    Num_Observations_Mat = np.stack(grouped['Num_Observations'].to_numpy())
    return Density_Mat, Flow_Mat, Speed_Mat, Num_Observations_Mat, time_bins, polar_x_bins, polar_y_bins
    