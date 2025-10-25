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
from dataclasses import dataclass, field
from typing import Tuple, Dict


# #############################################################################
# CONSTANTS: CRB dataset
# #############################################################################
@dataclass
class CRB_Config:
    data_root: str = "C:/Users/ShaimaaElBaklish/Documents/GitHub/bicycle_dataset/data/6_final_trajectories/"
    homography_root: str = "C:/Users/ShaimaaElBaklish/Documents/GitHub/bicycle_dataset/data/1_homography/circles/"
    video_path: str = "C:/Users/ShaimaaElBaklish/Desktop/Python_Workspace/bicycles/Videos/"
    videos: Tuple = ('DJI_20240906103036_0003_D.MP4', 'DJI_20240906103442_0004_D.MP4', 
              'DJI_20240906103850_0005_D.MP4', 'DJI_20240906104511_0007_D.MP4', 
              'DJI_20240906104917_0008_D.MP4', 'DJI_20240906105321_0009_D.MP4', 
              'DJI_20240906105621_0010_D.MP4', 'DJI_20240906110027_0011_D.MP4', 
              'DJI_20240906110432_0012_D.MP4')
    video_parts_X: Dict = field(default_factory=lambda: {
        'DJI_20240906103036_0003_D.MP4': ['PART_1', 'PART_2X', 'PART_3X', 'PART_4'],
        'DJI_20240906103442_0004_D.MP4': ['PART_1X', 'PART_2X'],
        'DJI_20240906103850_0005_D.MP4': ['PART_1X'],
        'DJI_20240906104511_0007_D.MP4': ['PART_1X', 'PART_23X'],
        'DJI_20240906104917_0008_D.MP4': ['PART_1X', 'PART_2X'],
        'DJI_20240906105321_0009_D.MP4': ['PART_1'],
        'DJI_20240906105621_0010_D.MP4': ['PART_1', 'PART_2', 'PART_3', 'PART_4', 'PART_5X', 'PART_6X'],
        'DJI_20240906110027_0011_D.MP4': ['PART_1X', 'PART_2345X'],
        'DJI_20240906110432_0012_D.MP4': ['PART_1']
    })
    video_lane_widths: Dict = field(default_factory=lambda: {
        'DJI_20240906103036_0003_D.MP4': 2.5,
        'DJI_20240906103442_0004_D.MP4': 2.5,
        'DJI_20240906103850_0005_D.MP4': 2.5,
        'DJI_20240906104511_0007_D.MP4': 2.5,
        'DJI_20240906104917_0008_D.MP4': 2.5,
        'DJI_20240906105321_0009_D.MP4': 2.5,
        'DJI_20240906105621_0010_D.MP4': 3.75,
        'DJI_20240906110027_0011_D.MP4': 3.75,
        'DJI_20240906110432_0012_D.MP4': 3.75,
    })
    sampling_freq: float = 25.0
    circle_outer_radius: float = 15.0


default_drawing_settings = {
    "homography": {
        "draw_line": True,
        "draw_fill": False,
        "draw_alpha": 0.5,
        "line_width": 5,
        "line_style": "--",
        "line_color": "cyan",
        "fill_color": "blue",
        "fill_hatch": None,
    },
    "region_of_interest": {
        "draw_line": True,
        "draw_fill": True,
        "draw_alpha": 0.1,
        "line_width": 5,
        "line_style": "--",
        "line_color": "cyan",
        "fill_color_in": "green",
        "fill_color_ex": "red",
        "fill_hatch": None,
    },
    "vehicle_annotations": {
        "draw_alpha": 0.5,
        "line_width": 1,
        "line_color": "red",
    },
    "labelled_vehicle_annnotations": {
        "draw_alpha": 0.5,
        "circle_radius": 100,
        "line_width": 5,
        "line_color": "white",
        "line_style": "--",
        "font_color": "white",
        "font_size": 20,
    },
    "history": {
        "history_skip": 10,
    }
}
default_drawing_settings["bicycle_colors"] = {
    'BICYCLE_1':   'red',
    'BICYCLE_2':   'blue',
    'BICYCLE_3':   'green',
    'BICYCLE_4':   'orange',
    'BICYCLE_5':   'purple',
    'BICYCLE_6':   'cyan',
    'BICYCLE_7':   'magenta',
    'BICYCLE_8':   'yellow',
    'BICYCLE_9':   'brown',
    'BICYCLE_10':  'lime',
    'BICYCLE_11':  'beige',
    'BICYCLE_12':  'pink',
    'BICYCLE_13':  'olive',
    'BICYCLE_14':  'teal',
    'BICYCLE_15':  'navy',
    'BICYCLE_16':  'gold',
    'BICYCLE_17':  'coral',
    'BICYCLE_18':  'orchid',
    'BICYCLE_19':  'turquoise',
    'BICYCLE_20':  'silver',
    'BICYCLE_21':  'indigo',
    'BICYCLE_23':  'maroon',
    'BICYCLE_24':  'slateblue',
    'BICYCLE_26':  'firebrick',
    'BICYCLE_27':  'grey',
    'BICYCLE_28':  'fuchsia'
}


# #############################################################################
# CONSTANTS: SRF dataset
# #############################################################################
@dataclass
class SRF_Config:
    data_root: str = "C:/Users/ShaimaaElBaklish/Documents/GitHub/trajectory_analysis/data_trajectories/7_final_trajectories_reconstructed/"
    video_path: str = "C:/Users/ShaimaaElBaklish/Desktop/Papers/ScientificReports_VehicleTrajectory_2025/"
    videos: Tuple = ("DJI_0933.MOV", "DJI_0934.MOV", "DJI_0939.MOV", "DJI_0940.MOV", "DJI_0943.MOV", "DJI_0944.MOV")
    videos_inner: Tuple = ("DJI_0939.MOV", "DJI_0940.MOV", "DJI_0943.MOV", "DJI_0944.MOV")
    videos_outer: Tuple = ("DJI_0933.MOV", "DJI_0934.MOV")
    sampling_freq: float = 25.0


# #############################################################################
# CONSTANTS: NGSIM dataset
# #############################################################################
@dataclass
class NGSIM_Config:
    data_root: str = "C:/Users/ShaimaaElBaklish/Documents/Datasets/NGSIM/"
    locations: Tuple[str] = ("Lankershim-Boulevard-LosAngeles-CA",
                             "Peachtree-Street-Atlanta-GA")
    traj_filenames: Tuple[str] = ("NGSIM__Lankershim_Vehicle_Trajectories.csv",
                                  "NGSIM_Peachtree_Vehicle_Trajectories.csv")
    sampling_freq: float = 10.0
    


