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
import cv2
import sys
import shutil
import warnings
warnings.simplefilter('ignore', RuntimeWarning) # Ignore all RuntimeWarnings
warnings.simplefilter('ignore', UserWarning) # Ignore all UserWarnings
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from tqdm import tqdm

from _constants import CRB_Config
CRB_Config = CRB_Config()
from _constants import default_drawing_settings as drawing_settings
from tools_data import compute_lane_coordinates
from tools_data import determine_leader, determine_leader_V2, determine_leader_hoogendoorn

# #############################################################################
# CONSTANTS
# #############################################################################
RELEVANT_VIDEO = 'DJI_20240906103036_0003_D.MP4'
RELEVANT_PART = "PART_2X"

# RELEVANT_VIDEO = 'DJI_20240906110027_0011_D.MP4'
# RELEVANT_PART = "PART_2345X"

video_file_path = CRB_Config.video_path + RELEVANT_VIDEO
zoom_factor = 3.5
history_horizon = 100

RELEVANT_FRAME_FROM = 2425 # Video 3, Part 2X
RELEVANT_FRAME_TO = 3720 # Video 3, Part 2X

# RELEVANT_FRAME_FROM = 2275 # Video 11, Part 2345X
# RELEVANT_FRAME_TO = 6120 # Video 11, Part 2345X

# #############################################################################
# METHODS
# #############################################################################
def loadHomography(homography_file: str):
    """
    This method loads the homography information for each frame. In this 
    project we employ the large circle in the middle of the circular road as 
    the characteristic pattern.

    Parameters
    ----------
    homography_file : str
        The path to the homography file.
        
    Returns
    -------
    df_circles: pd.DataFrame
        A dataframe, including the columns: "frame_nr", "x", "y", "r"
        This information can be used to transform the pixel coordinate system
        to Cartesian coordinates, knowing the real radius of the circle.
    """
    df_homography = []
    file = open(homography_file, "r")
    line = file.readline()
    while line!="":
        parts = line.replace("["," ").replace("]", " ").replace("\n", "").replace("\t", " ").split(" ")
        while "" in parts:
            parts.remove("")
        frame_nr = parts[0]
        x = parts[1]
        y = parts[2]
        r = parts[3]
        frame_nr = int(frame_nr)
        x = float(x)
        y = float(y)
        r = float(r)
        df_homography.append([frame_nr,x,y,r])
        line = file.readline()
    file.close()
    df_homography= pd.DataFrame(df_homography, columns=["frame_nr", "x", "y", "r"])
    return df_homography


def getNumberOfFramesFromVideo(video_file_path: str):
    """
    This method determines the number of frames in a given video file.

    Parameters
    ----------
    video_file_path : str
        The path to the video file.

    Returns
    -------
    num_frames : int
        The number of frames in the video file.
    """
    vidcap = cv2.VideoCapture(video_file_path)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    return num_frames

# #############################################################################
# LOADING - FILES
# #############################################################################
# VIDEO
num_frames =  getNumberOfFramesFromVideo(video_file_path)
# HOMOGRAPHY
df_homography = loadHomography(CRB_Config.homography_root+RELEVANT_VIDEO+"_homography3.txt")
# FINAL TRAJECTORY
df_final_trajectory = pd.read_csv(CRB_Config.data_root+RELEVANT_VIDEO+"_"+RELEVANT_PART+".txt")

# #############################################################################
# DEFINITION OF INPUT
# #############################################################################
elements = {
    "homography": df_homography,
    "labelled_final_trajectory": df_final_trajectory,
}


# #############################################################################
# RENDER VIDEO
# #############################################################################
df_final_trajectory = compute_lane_coordinates(df_final_trajectory)
# df_final_trajectory = determine_leader(df_final_trajectory)
df_final_trajectory = determine_leader_V2(df_final_trajectory)
# df_final_trajectory = determine_leader_hoogendoorn(df_final_trajectory)
x_c, y_c, r_c = 0, 0, CRB_Config.circle_outer_radius 

# Prepare output directory
frames_dir = 'frames_temp'
os.makedirs(frames_dir, exist_ok=True)

xlim = (-CRB_Config.circle_outer_radius - 2, CRB_Config.circle_outer_radius + 2)
ylim = (-CRB_Config.circle_outer_radius - 2, CRB_Config.circle_outer_radius + 2)
innerLaneWidth = 2.5
outerLaneWidth = 3.75

frames = df_final_trajectory['Frame_ID'].unique()
plt.ioff()

# Generate and save each frame as PNG
# check frames 2550 and 2580 and 2600 and 2650 and 2478, 3066, 3412
# frames2 = frames[frames >= 3600]
for i, frame_id in enumerate(tqdm(frames, desc="Saving frames")):
    frame_data = df_final_trajectory[df_final_trajectory['Frame_ID'] == frame_id]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f'Frame {frame_id}')
    ax.axis('off')
    
    for _, row in frame_data.iterrows():
        bike_id = row['Vehicle_ID'].split('_')[1]
        if pd.isna(row['Preceding']):
            prec_bike_id = "N/A"
        else:
            prec_bike_id = row['Preceding'].split('_')[1]
        color = drawing_settings["bicycle_colors"].get(row['Vehicle_ID'], 'gray')
        
        # Plot bike
        ax.scatter(row['Cartesian_X'], row['Cartesian_Y'], color=color, s=20)
        
        # Annotate with preceding_id
        label = f"ID:{bike_id}→{prec_bike_id}"
        ax.text(row['Cartesian_X'] - 1.0, row['Cartesian_Y'] + 0.5, label, fontsize=6, color='black')
    
    circle = plt.Circle(
        (x_c, y_c), r_c, 
        # general
        fill = False, alpha = 0.25, 
        # Line Specific
        edgecolor = "black", linestyle = "dotted", linewidth = 2.0,
    )
    ax.add_patch(circle)
    # circle = plt.Circle(
    #     (x_c, y_c), r_c-innerLaneWidth, 
    #     # general
    #     fill = False, alpha = 0.25, 
    #     # Line Specific
    #     edgecolor = "black", linestyle = "dotted", linewidth = 2.0,
    # )
    # ax.add_patch(circle)
    cicle = plt.Circle(
        (x_c, y_c), r_c-outerLaneWidth, 
        # general
        fill = False, alpha = 0.25, 
        # Line Specific
        edgecolor = "black", linestyle = "dotted", linewidth = 2.0,
    )
    ax.add_patch(cicle)
    ax.invert_yaxis()
    fig.tight_layout()
    
    # Save the figure to disk
    frame_filename = os.path.join(frames_dir, f'frame_{i:04d}.png')
    # plt.show()
    # sys.exit(1)
    
    plt.savefig(frame_filename, dpi=100)
    plt.close(fig)
    del fig, ax, circle
    gc.collect()

# Collect and save as GIF
images = []
for i in tqdm(range(len(frames)), desc="Creating GIF"):
    frame_filename = os.path.join(frames_dir, f'frame_{i:04d}.png')
    images.append(imageio.imread(frame_filename))

imageio.mimsave(os.path.join(CRB_Config.video_path, RELEVANT_VIDEO.split('.')[0]+"_"+RELEVANT_PART+"_gif.mp4"), images, fps=25)
# TODO: Add Hoogendoorn if using his method of leader-follower ID

# Optional: delete temporary frames folder
shutil.rmtree(frames_dir)

plt.ion()
# I need to call plt.show() now that I have turned off interactive backend
# To turn on interactive backend again, restart kernel

