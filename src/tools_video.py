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

import numpy as np
import pandas as pd

# #############################################################################
# CONSTANTS
# #############################################################################
CIRCLE_DIAMETER = 2 * 5.0

# #############################################################################
# METHODS: HOMOGRAPHY
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


def getFrameHomography(df_homography: pd.DataFrame, frame_nr: int):
    """
    This method loads the homography information for each frame. In this 
    project we employ the large circle in the middle of the circular road as 
    the characteristic pattern.

    Parameters
    ----------
    df_homography : pd.DataFrame
        The loaded homography dataframe.
    frame_nr: int
        The frame number that the homograhy should be loaded for.
        
    Returns
    -------
    frame_homography : list[int]
        The frame-specific homography list of three integers: "x", "y", "r"
    """
    df_selection = df_homography[df_homography["frame_nr"]==frame_nr]
    frame_homography = [df_selection["x"].tolist()[0], 
                        df_selection["y"].tolist()[0], 
                        df_selection["r"].tolist()[0]]
    return frame_homography


def transformPointFrom_PIX_2_CARTESIAN(point: list, frame_homography: list):
    """
    Transform point from pixel to Cartesian coordinates.

    Parameters
    ----------
    point : list[float]
        The coordinates of the point: "x", "y" (pixel coordinates)
    frame_homography: list[int]
        The homography list of three integers: "x", "y", "r"
        
    Returns
    -------
    point_new : list[float]
        The coordinates of the point: "x", "y" (Cartesian coordinates)
    """
    scale_factor_meter_per_pixel = CIRCLE_DIAMETER / (2*frame_homography[2])
    point_new = point.copy()
    point_new[0] =  (point[0] - frame_homography[0]) * scale_factor_meter_per_pixel
    point_new[1] =  (point[1] - frame_homography[1]) * scale_factor_meter_per_pixel
    return point_new

def transformPointFrom_CARTESIAN_2_PIX(point: list, frame_homography: list):
    """
    Transform point from Cartesian to pixel coordinates.

    Parameters
    ----------
    point : list[float]
        The coordinates of the point: "x", "y" (Cartesian coordinates)
    frame_homography: list[int]
        The homography list of three integers: "x", "y", "r"
        
    Returns
    -------
    point_new : list[float]
        The coordinates of the point: "x", "y" (pixel coordinates)
    """
    scale_factor_meter_per_pixel = CIRCLE_DIAMETER / (2*frame_homography[2])
    point_new = point.copy()
    point_new[0] = point[0] / scale_factor_meter_per_pixel + frame_homography[0]
    point_new[1] = point[1] / scale_factor_meter_per_pixel + frame_homography[1]
    return point_new

# #############################################################################
# METHODS: FRAME AND VIDEO PROCESSING
# #############################################################################
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


def extractFrameFromVideo(video_file_path: str, frame_nr: int):
    """
    This method loads a specific frame from a given video file.

    Parameters
    ----------
    video_file_path : str
        The path to the video file.
    frame_nr : int
        The frame number. First frame is 0.
        
    Returns
    -------
    success: bool
        Whether the loading was successful.
    frame : uint8 Array [HEIGHTxWIDTHx3]
        The frame as uint8 Array in RGB format.
    """
    try:
        vidcap = cv2.VideoCapture(video_file_path)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr-1)
        success, frame = vidcap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = None
        return success, frame
    except:
        return False, None

