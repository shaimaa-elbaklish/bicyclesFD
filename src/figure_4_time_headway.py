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
import gc
import warnings
warnings.simplefilter('ignore', RuntimeWarning) # Ignore all RuntimeWarnings
warnings.simplefilter('ignore', UserWarning) # Ignore all UserWarnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from scipy.integrate import cumulative_trapezoid
from sklearn.neighbors import KernelDensity

from _constants import CRB_Config
CRB_Config = CRB_Config()
from tools_data import compute_lane_coordinates
from tools_data import determine_leader, determine_leader_V2, determine_leader_Hoogendoorn

# #############################################################################
# MAIN: Proposed method of leader-follower pair ID (ALL Videos)
# #############################################################################
# all_time_headways = []
# for video in CRB_Config.videos:
#     print(f"... Processing video {video}.")
#     counter = 0
#     for part in CRB_Config.video_parts_X[video]:
#         df = pd.read_csv(CRB_Config.data_root + f"{video}_{part}.txt")
#         df = compute_lane_coordinates(df)
#         # df = determine_leader(df)
#         df = determine_leader_V2(df, polarX_max_threshold = 120*np.pi/180)
#         # df = determine_leader_Hoogendoorn(df)
#         df['Lane_Width'] = CRB_Config.video_lane_widths[video]
#         all_time_headways.append(df[['Time_Hdwy', 'Polar_X', 'Lane_Width']].to_numpy())
#         counter += 1
#         print(f"... Processed video part {part}. Finished {counter}/{len(CRB_Config.video_parts_X[video])}.")
# del df
# gc.collect()

# all_time_headways = np.concat(all_time_headways)
# np.save('../data/CRB_Time_Headways_ALLVideos_V2_Max120deg.npy', all_time_headways)
# # np.save('../data/CRB_Time_Headways_Hoogendoorn_ALLVideos_V2.npy', all_time_headways)
# sys.exit(1)

# #############################################################################
# FUNCTIONS: Solve Integral Equation Iteratively
# #############################################################################
def solve_iterative_grid_adaptive(h, f_vals, A, lam, 
                                  phi0=None, 
                                  max_iter=2000, 
                                  tol=1e-8,
                                  relax_min=0.05,
                                  relax_max=1.0,
                                  relax_factor=0.7,
                                  verbose=True):
    """
    Solve integral equation:
        r1(h) = (A*lam/phi) * exp(-lam*h) * ∫_0^h (f(τ) - r1(τ)) dτ
    s.t.    phi = ∫_0^∞ (f(s) - r1(s)) ds

    Parameters
    ----------
    h : ndarray
        Grid for h (1D, increasing).
    f_vals : ndarray
        Values of f(h) at the grid points.
    A, lam : floats
        Parameters in the equation.
    phi0 : float, optional
        Initial guess for phi. If None, computed from f - r1.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance (on residuals).
    relax_min, relax_max : floats
        Min/max relaxation factors.
    relax_factor : float
        Factor to reduce relaxation when residual increases.
    verbose : bool
        Print convergence messages.

    Returns
    -------
    phi : float
        Normalization constant.
    r1 : ndarray
        Approximated r1(h).
    history : dict
        Iteration history with residuals and relax values.
    """

    # initial guess: exponential form (not scaled by phi yet)
    r1 = A * lam * np.exp(-lam * h)

    # initial phi
    phi = phi0 if phi0 is not None else np.trapezoid(f_vals - r1, h)

    relax = 0.5
    history = {"iter": [], "res_r1": [], "res_phi": [], "relax": []}

    for it in range(max_iter):
        # compute cumulative integral of (f - r1) up to each h
        diff = f_vals - r1
        cum_int = cumulative_trapezoid(diff, h, initial=0)

        # compute updated r1
        r1_new = (A * lam / phi) * np.exp(-lam * h) * cum_int

        # updated phi
        phi_new = np.trapezoid(f_vals - r1_new, h)

        # residuals
        res_r1 = np.max(np.abs(r1_new - r1))
        res_phi = np.abs(phi_new - phi)

        # store history
        history["iter"].append(it)
        history["res_r1"].append(res_r1)
        history["res_phi"].append(res_phi)
        history["relax"].append(relax)

        # check convergence
        if res_r1 < tol and res_phi < tol:
            if verbose:
                print(f"Converged in {it} iterations.")
            return phi_new, r1_new, history

        # relaxation update
        if it > 0 and res_r1 > history["res_r1"][-2]:
            relax = max(relax * relax_factor, relax_min)
        else:
            relax = min(relax / relax_factor, relax_max)

        # relaxed update
        r1 = relax * r1_new + (1 - relax) * r1
        phi = relax * phi_new + (1 - relax) * phi
        
        if verbose and it % 2 == 0:
            print(f"Iteration {it}/{max_iter}: res_phi={res_phi:.2f} and res_r1={res_r1:.2f}")
    
    if verbose:
        print("Did not converge.")
    return phi, r1, history
    

def plot_history(history):
    iters = history["iter"]
    res_r1 = history["res_r1"]
    res_phi = history["res_phi"]
    relax_vals = history["relax"]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # Residual for r1
    axs[0].semilogy(iters, res_r1, marker="o")
    axs[0].set_title("Residual for r1")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("max |Δr1|")

    # Residual for phi
    axs[1].semilogy(iters, res_phi, marker="o", color="orange")
    axs[1].set_title("Residual for phi")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("|Δphi|")

    # Relaxation parameter
    axs[2].plot(iters, relax_vals, marker="o", color="green")
    axs[2].set_title("Relaxation parameter")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("relax")

    plt.tight_layout()
    plt.show()
    
# #############################################################################
# MAIN: Figure 4 (Time Headway Distributions for all videos)
#       Comparing Methods
# #############################################################################
files = [
    ('Proposed-30deg', '../data/CRB_Time_Headways_ALLVideos_V2_Max30deg.npy','Time_Headway_Survival_Function_Proposed_30deg'),
    ('Proposed-60deg', '../data/CRB_Time_Headways_ALLVideos_V2.npy', 'Time_Headway_Survival_Function_Proposed_60deg'),
    ('Proposed-90deg', '../data/CRB_Time_Headways_ALLVideos_V2_Max90deg.npy', 'Time_Headway_Survival_Function_Proposed_90deg'),
    ('Proposed-120deg', '../data/CRB_Time_Headways_ALLVideos_V2_Max120deg.npy', 'Time_Headway_Survival_Function_Proposed_120deg'),
    ('Hoogendoorn', '../data/CRB_Time_Headways_Hoogendoorn_ALLVideos_V2.npy', 'Time_Headway_Survival_Function_Hoogendoorn')
]
Hmax, dh = 50.0, 0.05
h = np.arange(0.0, Hmax+dh, dh)

lane_width_setting = 2.5
figsame, axsame = plt.subplots(1, 2, figsize=(8, 4))
for (label, path, save_name) in files:
    print(f"\n\n {label} METHOD")
    all_time_headways = np.load(path, allow_pickle=True)
    mask = all_time_headways[:, 2] == lane_width_setting
    th = all_time_headways[mask, 0]
    th = pd.Series(th).dropna().to_numpy()
    th = th.astype(np.float32)
    th = th[(th >= 0) & (th <= 50)]
    
    Tstar = 8.0
    # if 'Proposed' in label:
    #     Tstar = 10.0
    # elif label == 'Hoogendoorn':
    #     Tstar = 8.0
    # else:
    #     Tstar = 12.0
    
    n, m = th.shape[0], th[th > Tstar].shape[0]
    lmbda_hat = 1.0 / np.mean(th[th > Tstar])
    Ahat = m*np.exp(lmbda_hat*Tstar)/n
    kde = KernelDensity(kernel='epanechnikov', bandwidth=0.4).fit(th.reshape(-1, 1)) # OR tophat OR epanechnikov kernels
    f_vals = np.exp(kde.score_samples(h.reshape(-1, 1)))
    phi, r1, history = solve_iterative_grid_adaptive(h, f_vals, Ahat, lmbda_hat,
                                                     phi0=0.9, tol=1e-06, max_iter=10, verbose=False)
    tol = 1e-06
    r1_final = np.where(r1 <= -tol, 0, r1)
    g1_final = np.where(f_vals - r1_final <= -tol, 0, f_vals - r1_final)
    phi_final = np.trapezoid(g1_final, h)
    r1_final = np.round(r1_final, decimals=5)
    g1_final = np.round(g1_final, decimals=5)
    
    print("Final phi =", phi_final)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].hist(th, bins=1000, cumulative=-1, density=True, histtype='step')
    axs[0].axvline(x=Tstar, ymin=1e-04, ymax=10, color='red', linestyle='dashed', linewidth=1.5)
    axs[0].text(x=Tstar+0.5, y=5e-03, s='$T^{\\star}$', fontweight='bold', color='red', fontsize=12)
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Time Headway [s]')
    axs[0].set_ylabel('1 - CDF (log scale)')
    axs[0].set_xlim([0, 40])
    axs[0].set_ylim([5e-03, 1.5])

    axs[1].hist(th, bins=1000, density=True, histtype='step', label='Data')
    # axs[1].plot(h, f_vals, label='f(h)', linestyle='dashed', alpha=0.75)
    axs[1].plot(h, g1_final, label='$g_1(h)$, Constrained Headway', alpha=0.65, linestyle='dashed')
    axs[1].plot(h, r1_final, label='$r_1(h)$, Free Headway', alpha=0.65, linestyle='dashed')
    axs[1].set_xlabel('Time Headway [s]')
    axs[1].set_ylabel('PDF')
    axs[1].set_xlim([0, 40])
    axs[1].set_ylim([0, 0.6])
    axs[1].legend(loc='upper right')

    fig.tight_layout()
    # ZOOM IN
    pos = axs[1].get_position()
    inset_width = 0.4 * pos.width
    inset_height = 0.4 * pos.height
    inset_left = pos.x0 + 0.55*pos.width  # adjust fraction inside parent
    inset_bottom = pos.y0 + 0.25*pos.height
    ax_inset = fig.add_axes([inset_left, inset_bottom, inset_width, inset_height])
    ax_inset.hist(th, bins=1000, density=True, histtype='step', label='Data')
    ax_inset.plot(h, g1_final, label='$g_1(h)$, Constrained Headway', alpha=0.65, linestyle='dashed')
    ax_inset.plot(h, r1_final, label='$r_1(h)$, Free Headway', alpha=0.65, linestyle='dashed')
    x1, x2 = 0, 40
    y1, y2 = 0, 0.004
    if 'Proposed' in label:
        y1, y2 = 0, 0.004
    elif label == 'Hoogendoorn':
        y1, y2 = 0, 0.025
    ax_inset.set_xlim([x1, x2])
    ax_inset.set_ylim([y1, y2])
    y2 += 0.01
    rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='black', facecolor='none', linestyle='--')
    axs[1].add_patch(rect)

    rect_corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    # inset axis corners in figure coords
    inset_corners = [
        (inset_left, inset_bottom),
        (inset_left+inset_width, inset_bottom),
        (inset_left, inset_bottom+inset_height),
        (inset_left+inset_width, inset_bottom+inset_height)
    ]

    # transform rect corners from data coords -> figure coords
    trans = axs[1].transData.transform
    rect_fig = fig.transFigure.inverted().transform(trans(rect_corners))

    # pick which corners to connect (e.g. bottom-left, top-right)
    pairs = [(2,0), (3,1)]  # (rect_corner_index, inset_corner_index)

    for rc, ic in pairs:
        line = Line2D([rect_fig[rc,0], inset_corners[ic][0]],
                      [rect_fig[rc,1], inset_corners[ic][1]],
                      transform=fig.transFigure, color="black", linestyle="--", linewidth=1)
        fig.add_artist(line)

    theta_vals = g1_final / f_vals # Eq. (13)
    ub_idx = np.argmax(theta_vals <= 0) + 1 # np.argmax(h >= 20)
    p = np.polyfit(h[:ub_idx], theta_vals[:ub_idx], 2)
    theta_poly = np.polyval(p, h[:ub_idx])
    axs[2].plot(h[:ub_idx], theta_vals[:ub_idx], label='Data', linestyle='dashed')
    axs[2].plot(h[:ub_idx], theta_poly, label='Quadratic Fit')
    axs[2].set_xlabel('Time Headway [s]')
    axs[2].set_ylabel('Conditional Probability of Following')
    axs[2].legend(loc='upper right')

    fig.savefig(f'../figures/{save_name}.pdf', dpi=300, bbox_inches='tight')


    EX = np.sum(h * g1_final/phi_final * dh)
    VarX = np.sum((h - EX)**2 * g1_final/phi_final * dh)
    capacity = 3600 / EX
    print(f"{label}: expected following time headway = {EX:.2f} seconds")
    print(f"{label}: STD following time headway = {np.sqrt(VarX):.2f} seconds")
    print(f"{label}: Capacity = {capacity:.2f} bic/h")
    print(f"{label}, 2.5 Lane Width: Capacity = {3600/(2.5*EX):.2f} bic/h/m")
    print(f"{label}, 3.75 Lane Width: Capacity = {3600/(3.75*EX):.2f} bic/h/m")

    if label == 'Hoogendoorn':
        axsame[0].plot(h, g1_final/phi_final, label=f'{label}', alpha=0.6, linestyle='--')
        axsame[1].plot(h, r1_final/(1-phi_final), label=f'{label}', alpha=0.6, linestyle='--')
    else:
        axsame[0].plot(h, g1_final/phi_final, label=f'${label}', alpha=0.6)
        axsame[1].plot(h, r1_final/(1-phi_final), label=f'{label}', alpha=0.6)


axsame[0].legend(loc='upper right')
axsame[0].set_xlim([0, 12])
axsame[0].set_xlabel('Time Headway [s]')
axsame[0].set_ylabel('PDF $g(h)$')
axsame[1].set_xlabel('Time Headway [s]')
axsame[1].set_ylabel('PDF $r(h)$')
figsame.tight_layout()
figsame.savefig('../figures/Time_Headway_Distributions_Comparison.pdf', dpi=300, bbox_inches='tight')

sys.exit(1)

# #############################################################################
# MAIN: Figure 4 (Time Headway Distributions for all videos)
#       Using Proposed Method
# #############################################################################
all_time_headways = np.load('../data/CRB_Time_Headways_ALLVideos_V2.npy', allow_pickle=True)
th = all_time_headways[:, 0]
th = pd.Series(th).dropna().to_numpy()
th = th.astype(np.float32)
th = th[(th >= 0) & (th <= 50)]

# Hoogendoorn and Daamen (2016), Eq. (7)--(11)

Tstar = 10.0
n, m = th.shape[0], th[th > Tstar].shape[0]
lmbda_hat = 1.0 / np.mean(th[th > Tstar])
Ahat = m*np.exp(lmbda_hat*Tstar)/n

Hmax, dh = 50.0, 0.01
h = np.arange(0.0, Hmax+dh, dh)

kde = KernelDensity(kernel='epanechnikov', bandwidth=0.4).fit(th.reshape(-1, 1)) # OR tophat OR epanechnikov kernels
f_vals = np.exp(kde.score_samples(h.reshape(-1, 1)))

phi, r1, history = solve_iterative_grid_adaptive(h, f_vals, Ahat, lmbda_hat,
                                                 phi0=0.9, tol=1e-4, max_iter=10, verbose=True)
# plot_history(history)

tol = 1e-06
r1_final = np.where(r1 <= -tol, 0, r1)
g1_final = np.where(f_vals - r1_final <= -tol, 0, f_vals - r1_final)
phi_final = np.trapezoid(g1_final, h)
r1_final = np.round(r1_final, decimals=5)
g1_final = np.round(g1_final, decimals=5)

print("Final phi =", phi_final)
print("Mass r1 =", np.trapezoid(r1_final, h))

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].hist(th, bins=1000, cumulative=-1, density=True, histtype='step')
axs[0].axvline(x=Tstar, ymin=1e-04, ymax=10, color='red', linestyle='dashed', linewidth=1.5)
axs[0].text(x=Tstar+0.5, y=5e-03, s='$T^{\\star}$', fontweight='bold', color='red', fontsize=12)
axs[0].set_yscale('log')
axs[0].set_xlabel('Time Headway [s]')
axs[0].set_ylabel('1 - CDF (log scale)')
axs[0].set_xlim([0, 40])
axs[0].set_ylim([5e-03, 1.5])

axs[1].hist(th, bins=1000, density=True, histtype='step', label='Data')
# axs[1].plot(h, f_vals, label='f(h)', linestyle='dashed', alpha=0.75)
axs[1].plot(h, g1_final, label='$g_1(h)$, Constrained Headway', alpha=0.65, linestyle='dashed')
axs[1].plot(h, r1_final, label='$r_1(h)$, Free Headway', alpha=0.65, linestyle='dashed')
axs[1].set_xlabel('Time Headway [s]')
axs[1].set_ylabel('PDF')
axs[1].set_xlim([0, 40])
axs[1].set_ylim([0, 0.6])
axs[1].legend(loc='upper right')

fig.tight_layout()
# ZOOM IN
pos = axs[1].get_position()
inset_width = 0.4 * pos.width
inset_height = 0.4 * pos.height
inset_left = pos.x0 + 0.55*pos.width  # adjust fraction inside parent
inset_bottom = pos.y0 + 0.25*pos.height
ax_inset = fig.add_axes([inset_left, inset_bottom, inset_width, inset_height])
ax_inset.hist(th, bins=1000, density=True, histtype='step', label='Data')
ax_inset.plot(h, g1_final, label='$g_1(h)$, Constrained Headway', alpha=0.65, linestyle='dashed')
ax_inset.plot(h, r1_final, label='$r_1(h)$, Free Headway', alpha=0.65, linestyle='dashed')
x1, x2 = 0, 40
y1, y2 = 0, 0.004
ax_inset.set_xlim([x1, x2])
ax_inset.set_ylim([y1, y2])
y2 += 0.01
rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='black', facecolor='none', linestyle='--')
axs[1].add_patch(rect)

rect_corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
# inset axis corners in figure coords
inset_corners = [
    (inset_left, inset_bottom),
    (inset_left+inset_width, inset_bottom),
    (inset_left, inset_bottom+inset_height),
    (inset_left+inset_width, inset_bottom+inset_height)
]

# transform rect corners from data coords -> figure coords
trans = axs[1].transData.transform
rect_fig = fig.transFigure.inverted().transform(trans(rect_corners))

# pick which corners to connect (e.g. bottom-left, top-right)
pairs = [(2,0), (3,1)]  # (rect_corner_index, inset_corner_index)

for rc, ic in pairs:
    line = Line2D([rect_fig[rc,0], inset_corners[ic][0]],
                  [rect_fig[rc,1], inset_corners[ic][1]],
                  transform=fig.transFigure, color="black", linestyle="--", linewidth=1)
    fig.add_artist(line)

theta_vals = g1_final / f_vals # Eq. (13)
ub_idx = np.argmax(theta_vals <= 0) + 1 # np.argmax(h >= 20)
p = np.polyfit(h[:ub_idx], theta_vals[:ub_idx], 2)
theta_poly = np.polyval(p, h[:ub_idx])
axs[2].plot(h[:ub_idx], theta_vals[:ub_idx], label='Data', linestyle='dashed')
axs[2].plot(h[:ub_idx], theta_poly, label='Quadratic Fit')
axs[2].set_xlabel('Time Headway [s]')
axs[2].set_ylabel('Conditional Probability of Following')
axs[2].legend(loc='upper right')

fig.savefig('../figures/Time_Headway_Survival_Function.pdf', dpi=300, bbox_inches='tight')


EX = np.sum(h * g1_final/phi_final * dh)
VarX = np.sum((h - EX)**2 * g1_final/phi_final * dh)
a = 0.8
capacity = 1 / (EX/3600)
print(f"Proposed: expected following time headway = {EX:.2f} seconds")
print(f"Proposed: STD following time headway = {np.sqrt(VarX):.2f} seconds")
print(f"Proposed: Capacity = {capacity:.2f} bic/h")
print(f"Proposed, 2.5 Lane Width: Capacity = {1/(EX/3600)/2.5:.2f} bic/h/m")
print(f"Proposed, 3.75 Lane Width: Capacity = {1/(EX/3600)/3.75:.2f} bic/h/m")
# Proposed: Qc = 1297.54 bic/h/m
# Hoogendoorn: Qc = 3738.99 bic/h/m

figsame = plt.figure(figsize=(4, 4))
axsame = plt.gca()
axsame.plot(h, g1_final/phi_final, label='$g(h)$, Proposed', color='tab:blue', alpha=0.6)
axsame.plot(h, r1_final/(1-phi_final), label='$r(h)$, Proposed', color='tab:orange', alpha=0.6)

# #############################################################################
# MAIN: Figure 4 (Time Headway Distributions for all videos)
#       Using Hoogendoorn's Method
# #############################################################################
del f_vals, r1, phi, h, th, all_time_headways
all_time_headways = np.load('../data/CRB_Time_Headways_Hoogendoorn_ALLVideos_V2.npy', allow_pickle=True)
th = all_time_headways[:, 0]
th = pd.Series(th).dropna().to_numpy()
th = th.astype(np.float32)
th = th[(th >= 0) & (th <= 50)]

# Hoogendoorn and Daamen (2016), Eq. (7)--(11)

Tstar = 8.0
n, m = th.shape[0], th[th > Tstar].shape[0]
lmbda_hat = 1.0 / np.mean(th[th > Tstar])
Ahat = m*np.exp(lmbda_hat*Tstar)/n

Hmax, dh = 50.0, 0.01
h = np.arange(0.0, Hmax+dh, dh)

kde = KernelDensity(kernel='epanechnikov', bandwidth=0.4).fit(th.reshape(-1, 1)) # OR tophat OR epanechnikov kernels
f_vals = np.exp(kde.score_samples(h.reshape(-1, 1)))

phi, r1, history = solve_iterative_grid_adaptive(h, f_vals, Ahat, lmbda_hat,
                                                 phi0=0.9, tol=1e-4, max_iter=10, verbose=True)
# plot_history(history)

tol = 1e-06
r1_final = np.where(r1 <= -tol, 0, r1)
g1_final = np.where(f_vals - r1_final <= -tol, 0, f_vals - r1_final)
phi_final = np.trapezoid(g1_final, h)
r1_final = np.round(r1_final, decimals=5)
g1_final = np.round(g1_final, decimals=5)

print("Final phi =", phi_final)
print("Mass r1 =", np.trapezoid(r1_final, h))

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].hist(th, bins=1000, cumulative=-1, density=True, histtype='step')
axs[0].axvline(x=Tstar, ymin=1e-04, ymax=10, color='red', linestyle='dashed', linewidth=1.5)
axs[0].text(x=Tstar+0.5, y=5e-03, s='$T^{\\star}$', fontweight='bold', color='red', fontsize=12)
axs[0].set_yscale('log')
axs[0].set_xlabel('Time Headway [s]')
axs[0].set_ylabel('1 - CDF (log scale)')
axs[0].set_xlim([0, 40])
axs[0].set_ylim([5e-03, 1.5])

axs[1].hist(th, bins=1000, density=True, histtype='step', label='Data')
# axs[1].plot(h, f_vals, label='f(h)', linestyle='dashed', alpha=0.75)
axs[1].plot(h, g1_final, label='$g_1(h)$, Constrained Headway', alpha=0.65, linestyle='dashed')
axs[1].plot(h, r1_final, label='$r_1(h)$, Free Headway', alpha=0.65, linestyle='dashed')
axs[1].set_xlabel('Time Headway [s]')
axs[1].set_ylabel('PDF')
axs[1].set_xlim([0, 40])
axs[1].set_ylim([0, 0.6])
axs[1].legend(loc='upper right')

fig.tight_layout()
# ZOOM IN
pos = axs[1].get_position()
inset_width = 0.4 * pos.width
inset_height = 0.4 * pos.height
inset_left = pos.x0 + 0.55*pos.width  # adjust fraction inside parent
inset_bottom = pos.y0 + 0.25*pos.height
ax_inset = fig.add_axes([inset_left, inset_bottom, inset_width, inset_height])
ax_inset.hist(th, bins=1000, density=True, histtype='step', label='Data')
ax_inset.plot(h, g1_final, label='$g_1(h)$, Constrained Headway', alpha=0.65, linestyle='dashed')
ax_inset.plot(h, r1_final, label='$r_1(h)$, Free Headway', alpha=0.65, linestyle='dashed')
x1, x2 = 0, 40
y1, y2 = 0, 0.025
ax_inset.set_xlim([x1, x2])
ax_inset.set_ylim([y1, y2])
y2 += 0.01
rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='black', facecolor='none', linestyle='--')
axs[1].add_patch(rect)

rect_corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
# inset axis corners in figure coords
inset_corners = [
    (inset_left, inset_bottom),
    (inset_left+inset_width, inset_bottom),
    (inset_left, inset_bottom+inset_height),
    (inset_left+inset_width, inset_bottom+inset_height)
]

# transform rect corners from data coords -> figure coords
trans = axs[1].transData.transform
rect_fig = fig.transFigure.inverted().transform(trans(rect_corners))

# pick which corners to connect (e.g. bottom-left, top-right)
pairs = [(2,0), (3,1)]  # (rect_corner_index, inset_corner_index)

for rc, ic in pairs:
    line = Line2D([rect_fig[rc,0], inset_corners[ic][0]],
                  [rect_fig[rc,1], inset_corners[ic][1]],
                  transform=fig.transFigure, color="black", linestyle="--", linewidth=1)
    fig.add_artist(line)

theta_vals = g1_final / f_vals # Eq. (13)
ub_idx = np.argmax(theta_vals <= 0) + 1 # np.argmax(h >= 20)
p = np.polyfit(h[:ub_idx], theta_vals[:ub_idx], 2)
theta_poly = np.polyval(p, h[:ub_idx])
axs[2].plot(h[:ub_idx], theta_vals[:ub_idx], label='Data', linestyle='dashed')
axs[2].plot(h[:ub_idx], theta_poly, label='Quadratic Fit')
axs[2].set_xlabel('Time Headway [s]')
axs[2].set_ylabel('Conditional Probability of Following')
axs[2].legend(loc='upper right')

fig.savefig('../figures/Time_Headway_Hoogendoorn_Survival_Function.pdf', dpi=300, bbox_inches='tight')

EX = np.sum(h * g1_final/phi_final * dh)
VarX = np.sum(np.square(h - EX) * g1_final/phi_final * dh)
a = 0.8
capacity = 1 / (EX/3600)
print(f"Proposed: expected following time headway = {EX:.2f} seconds")
print(f"Proposed: STD following time headway = {np.sqrt(VarX):.2f} seconds")
print(f"Hoogendoorn: Capacity = {capacity:.2f} bic/h")
print(f"Hoogendoorn, 2.5 Lane Width: Capacity = {1/(EX/3600)/2.5:.2f} bic/h/m")
print(f"Hoogendoorn, 3.75 Lane Width: Capacity = {1/(EX/3600)/3.75:.2f} bic/h/m")
# Proposed: Qc = 1297.54 bic/h/m
# Hoogendoorn: Qc = 3738.99 bic/h/m

axsame.plot(h, g1_final/phi_final, label='$g(h)$, Hoogendoorn', color='tab:blue', linestyle='dashed')
axsame.plot(h, r1_final/(1-phi_final), label='$r(h)$, Hoogendoorn', color='tab:orange', linestyle='dashed')
axsame.legend(loc='upper right', ncol=1, fontsize='medium')
figsame.tight_layout()
