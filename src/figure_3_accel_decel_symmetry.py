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
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import gridspec
from outliers import smirnov_grubbs as grubbs

from _constants import CRB_Config
CRB_Config = CRB_Config()

# #############################################################################
# MAIN: Load data file
# #############################################################################
video = 'DJI_20240906103036_0003_D.MP4'
part = 'PART_2X'
df = pd.read_csv(CRB_Config.data_root + f"{video}_{part}.txt")

# #############################################################################
# MAIN: Acceleration-Deceleration symmetry (single video)
# #############################################################################
grouped = df.groupby(by='Vehicle_ID')
df['v_Accel'] = grouped['v_Vel'].diff(1).shift(-1).fillna(0) * CRB_Config.sampling_freq

df_accel = df[df['v_Accel'] >= 0]
df_decel = df[df['v_Accel'] <= 0]
df_decel['v_Decel'] = df_decel['v_Accel'].abs()

# remove outliers for accelerations
outlier_idx = []
unique_bicycles = df['Vehicle_ID'].unique()
for bike_id in unique_bicycles:
    subdf = df[df['Vehicle_ID'] == bike_id].copy()
    outs = grubbs.two_sided_test_outliers(subdf['v_Accel'].to_numpy(), alpha=0.05)
    for out in outs:
        outlier_idx.append(subdf[subdf['v_Accel'] == out].index.values[0])

df_filtered = df.drop(index=outlier_idx)
df_accel_filtered = df_filtered[df_filtered['v_Accel'] >= 0]
df_decel_filtered = df_filtered[df_filtered['v_Accel'] <= 0]
df_decel_filtered['v_Decel'] = df_decel_filtered['v_Accel'].abs()

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
df_accel['v_Accel'].hist(ax=axs[0], bins=100, cumulative=1, density=True, histtype='bar', alpha=0.5, label='Acceleration')
df_decel['v_Decel'].hist(ax=axs[0], bins=100, cumulative=1, density=True, histtype='bar', alpha=0.5, label='Deceleration')
axs[0].set_xlabel('Acceleration Magnitude [m/s$^2$]')
axs[0].set_ylabel('CDF')
axs[0].legend()
axs[0].set_title('Original', fontweight='bold')

df_accel_filtered['v_Accel'].hist(ax=axs[1], bins=100, cumulative=1, density=True, histtype='bar', alpha=0.5, label='Acceleration')
df_decel_filtered['v_Decel'].hist(ax=axs[1], bins=100, cumulative=1, density=True, histtype='bar', alpha=0.5, label='Deceleration')
axs[1].set_xlabel('Acceleration Magnitude [m/s$^2$]')
axs[1].set_ylabel('CDF')
axs[1].set_title('Outliers Removed', fontweight='bold')
fig.tight_layout()
# fig.savefig(f'../figures/Accel_CDFhist_{video}_{part}.png', dpi=300, bbox_inches='tight')


fig = plt.figure(figsize=(8, 8))
ax0 = fig.add_subplot(211, frameon=False)
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_title("Original", fontweight='bold')
ax1 = fig.add_subplot(212, frameon=False)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title("Outliers Removed", fontweight='bold')

ax00 = fig.add_subplot(221, frameon=True)
sns.violinplot(data=df_accel, x=None, y='v_Accel', ax=ax00, cut=0, label='Acceleration')
sns.violinplot(data=df_decel, x=None, y='v_Accel', ax=ax00, cut=0, label='Deceleration')
ax00.set_ylabel('Acceleration [m/s$^2$]')
ax00.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax00.legend()
ax01 = fig.add_subplot(222, frameon=True)
sns.violinplot(data=df_accel, x='Vehicle_ID', y='v_Accel', ax=ax01, cut=0)
sns.violinplot(data=df_decel, x='Vehicle_ID', y='v_Accel', ax=ax01, cut=0)
ax01.set_ylabel('Acceleration [m/s$^2$]')
ax01.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax01.set_xlabel('')

ax10 = fig.add_subplot(223, frameon=True)
sns.violinplot(data=df_accel_filtered, x=None, y='v_Accel', ax=ax10, cut=0, label='Acceleration')
sns.violinplot(data=df_decel_filtered, x=None, y='v_Accel', ax=ax10, cut=0, label='Deceleration')
ax10.set_ylabel('Acceleration [m/s$^2$]')
ax10.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax10.legend()
ax11 = fig.add_subplot(224, frameon=True)
sns.violinplot(data=df_accel_filtered, x='Vehicle_ID', y='v_Accel', ax=ax11, cut=0)
sns.violinplot(data=df_decel_filtered, x='Vehicle_ID', y='v_Accel', ax=ax11, cut=0)
ax11.set_ylabel('Acceleration [m/s$^2$]')
ax11.set_xticks(ax11.get_xticks(), ax11.get_xticklabels(), 
                     rotation=45, ha="right", rotation_mode="anchor")
ax11.set_xlabel('')

fig.tight_layout()
# fig.savefig(f'../figures/Accel_PDFviolin_{video}_{part}.png', dpi=300, bbox_inches='tight')

# #############################################################################
# MAIN: Acceleration-Deceleration symmetry (ALL VIDEOS)
# #############################################################################
df_all = None
for video in CRB_Config.videos:
    for part in CRB_Config.video_parts_X[video]:
        df = pd.read_csv(CRB_Config.data_root + f"{video}_{part}.txt")
        grouped = df.groupby(by='Vehicle_ID')
        df['v_Accel'] = grouped['v_Vel'].diff(1).shift(-1).fillna(0) * CRB_Config.sampling_freq
        if df_all is None:
            df_all = df.copy()
        else:
            df_all = pd.concat((df_all, df), ignore_index=True)
del df
# order bicycles by ID
df_all['Vehicle_ID_Split'] = df_all['Vehicle_ID'].str.split('_')
df_all['Vehicle_Num'] = df_all['Vehicle_ID_Split'].apply(lambda x: int(x[1]))
df_all = df_all.sort_values(by=['Vehicle_Num', 'Frame_ID'], ascending=True)
df_all = df_all.drop(columns=['Vehicle_ID_Split'])

df_accel = df_all[df_all['v_Accel'] >= 0]
df_decel = df_all[df_all['v_Accel'] <= 0]
df_decel['v_Decel'] = df_decel['v_Accel'].abs()

# remove outliers for accelerations
outlier_idx = []
unique_bicycles = df_all['Vehicle_ID'].unique()
for bike_id in unique_bicycles:
    subdf = df_all[df_all['Vehicle_ID'] == bike_id].copy()
    outs = grubbs.two_sided_test_outliers(subdf['v_Accel'].to_numpy(), alpha=0.05)
    for out in outs:
        outlier_idx.append(subdf[subdf['v_Accel'] == out].index.values[0])
del subdf

df_filtered = df_all.drop(index=outlier_idx)
df_accel_filtered = df_filtered[df_filtered['v_Accel'] >= 0]
df_decel_filtered = df_filtered[df_filtered['v_Accel'] <= 0]
df_decel_filtered['v_Decel'] = df_decel_filtered['v_Accel'].abs()

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
df_accel['v_Accel'].hist(ax=axs[0], bins=100, cumulative=1, density=True, histtype='bar', alpha=0.5, label='Acceleration')
df_decel['v_Decel'].hist(ax=axs[0], bins=100, cumulative=1, density=True, histtype='bar', alpha=0.5, label='Deceleration')
axs[0].set_xlabel('Acceleration Magnitude [m/s$^2$]')
axs[0].set_ylabel('CDF')
axs[0].legend()
axs[0].set_title('Original', fontweight='bold')

df_accel_filtered['v_Accel'].hist(ax=axs[1], bins=100, cumulative=1, density=True, histtype='bar', alpha=0.5, label='Acceleration')
df_decel_filtered['v_Decel'].hist(ax=axs[1], bins=100, cumulative=1, density=True, histtype='bar', alpha=0.5, label='Deceleration')
axs[1].set_xlabel('Acceleration Magnitude [m/s$^2$]')
axs[1].set_ylabel('CDF')
axs[1].set_title('Outliers Removed', fontweight='bold')
fig.tight_layout()
# fig.savefig('../figures/Accel_CDFhist_ALLvideos.png', dpi=300, bbox_inches='tight')
fig.savefig('../figures/Accel_CDFhist_ALLvideos.pdf', dpi=300, bbox_inches='tight')


fig = plt.figure(figsize=(12, 8))
ax0 = fig.add_subplot(211, frameon=False)
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_title("Original", fontweight='bold')
ax1 = fig.add_subplot(212, frameon=False)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title("Outliers Removed", fontweight='bold')

gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3])

ax00 = fig.add_subplot(gs[0,0], frameon=True)
sns.violinplot(data=df_accel, x=None, y='v_Accel', ax=ax00, cut=0, label='Acceleration')
sns.violinplot(data=df_decel, x=None, y='v_Accel', ax=ax00, cut=0, label='Deceleration')
ax00.set_ylabel('Acceleration [m/s$^2$]')
ax00.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax00.legend(loc='upper left')
ax01 = fig.add_subplot(gs[0,1], frameon=True)
sns.violinplot(data=df_accel, x='Vehicle_ID', y='v_Accel', ax=ax01, cut=0)
sns.violinplot(data=df_decel, x='Vehicle_ID', y='v_Accel', ax=ax01, cut=0)
ax01.set_ylabel('Acceleration [m/s$^2$]')
ax01.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax01.set_xlabel('')

ax10 = fig.add_subplot(gs[1,0], frameon=True)
sns.violinplot(data=df_accel_filtered, x=None, y='v_Accel', ax=ax10, cut=0, label='Acceleration')
sns.violinplot(data=df_decel_filtered, x=None, y='v_Accel', ax=ax10, cut=0, label='Deceleration')
ax10.set_ylabel('Acceleration [m/s$^2$]')
ax10.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax10.legend(loc='upper left')
ax11 = fig.add_subplot(gs[1,1], frameon=True)
sns.violinplot(data=df_accel_filtered, x='Vehicle_ID', y='v_Accel', ax=ax11, cut=0)
sns.violinplot(data=df_decel_filtered, x='Vehicle_ID', y='v_Accel', ax=ax11, cut=0)
ax11.set_ylabel('Acceleration [m/s$^2$]')
ax11.set_xticks(ax11.get_xticks(), ax11.get_xticklabels(), 
                     rotation=45, ha="right", rotation_mode="anchor")
ax11.set_xlabel('')

fig.tight_layout()
# fig.savefig('../figures/Accel_PDFviolin_ALLvideos.png', dpi=300, bbox_inches='tight')
fig.savefig('../figures/Accel_PDFviolin_ALLvideos.pdf', dpi=300, bbox_inches='tight')


fig = plt.figure(figsize=(12, 4))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3])

ax0 = fig.add_subplot(gs[0], frameon=True)
sns.violinplot(data=df_accel_filtered, x=None, y='v_Accel', ax=ax0, cut=0, label='Acceleration')
sns.violinplot(data=df_decel_filtered, x=None, y='v_Accel', ax=ax0, cut=0, label='Deceleration')
ax0.set_ylabel('Acceleration [m/s$^2$]')
ax0.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
# ax0.set_ylim([-4, 4])
ax0.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25))
ax1 = fig.add_subplot(gs[1], frameon=True)
sns.violinplot(data=df_accel_filtered, x='Vehicle_ID', y='v_Accel', ax=ax1, cut=0)
sns.violinplot(data=df_decel_filtered, x='Vehicle_ID', y='v_Accel', ax=ax1, cut=0)
ax1.set_ylabel('Acceleration [m/s$^2$]')
ax1.set_xticks(ax11.get_xticks(), ax11.get_xticklabels(), 
                     rotation=45, ha="right", rotation_mode="anchor")
ax1.set_xlabel('')
# ax1.set_ylim([-4, 4])

fig.tight_layout()
# fig.savefig('../figures/Accel_PDFviolin_ALLvideos_NoOutliers.png', dpi=300, bbox_inches='tight')
fig.savefig('../figures/Accel_PDFviolin_ALLvideos_NoOutliers.pdf', dpi=300, bbox_inches='tight')
