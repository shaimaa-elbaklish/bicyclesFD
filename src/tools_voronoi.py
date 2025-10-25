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
import sys
import pathlib
import warnings
warnings.simplefilter('ignore', RuntimeWarning) # Ignore all RuntimeWarnings
warnings.simplefilter('ignore', UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon
from pedpy import PEDPY_BLUE, PEDPY_ORANGE, DENSITY_COL
from pedpy import load_trajectory, plot_trajectories, plot_measurement_setup
from pedpy import WalkableArea, MeasurementArea, Cutoff
from pedpy import compute_individual_voronoi_polygons, compute_voronoi_density
from pedpy import compute_individual_speed, compute_voronoi_speed
from pedpy import SpeedCalculation, plot_speed, plot_density, plot_voronoi_cells

# #############################################################################
# FUNCTIONS
# #############################################################################
def prepare_data_pedpy(df, lane_width, config):
    df = df[df["Polar_Y"] >= config.circle_outer_radius - lane_width]
    df = df[df["Polar_Y"] <= config.circle_outer_radius]
    df = df[["Vehicle_ID", "Frame_ID", "Cartesian_X", "Cartesian_Y"]]
    df = df.rename(columns={
        "Frame_ID": "frame",
        "Cartesian_X": "X",
        "Cartesian_Y": "Y"
    })
    df["ID"] = df["Vehicle_ID"].str.split("_")
    df["ID"] = df["ID"].apply(lambda x: x[-1])
    df = df.astype({"ID": "int"})
    df =  df.drop(columns=["Vehicle_ID"])
    df["Z"] = 1.1 # average bicycle height
    df = df[["ID", "frame", "X", "Y", "Z"]]
    return df


def filter_traj_inside(traj, walkable_area_poly):
    mask = traj.apply(lambda row: walkable_area_poly.contains(Point(row["x"], row["y"])), axis=1)
    return traj[mask]


def define_measurement_setup(num_sectors, lane_width):
    center_point = Point(0, 0)
    outer_radius, inner_radius = 15.1, 15.0 - lane_width
    outer = center_point.buffer(outer_radius, quad_segs=64)
    inner = center_point.buffer(inner_radius, quad_segs=64)
    #walkable_area_poly = outer.difference(inner)
    walkable_area_poly = Polygon(outer.exterior.coords, holes=[inner.exterior.coords])
    walkable_area = WalkableArea(walkable_area_poly)
    
    r_max = 20
    thetas = np.linspace(0, 2*np.pi, num_sectors+1)
    measurement_areas = []
    for i in range(len(thetas)-1):
        theta1, theta2 = thetas[i], thetas[i+1]
        wedge = Polygon([(inner_radius * np.cos(theta1), inner_radius * np.sin(theta1)),
                         (inner_radius * np.cos(theta2), inner_radius * np.sin(theta2)),
                         (r_max * np.cos(theta2), r_max * np.sin(theta2)), 
                         (r_max * np.cos(theta1), r_max * np.sin(theta1))])
        # intersection to get measurement area as a convex set
        measurement_area_poly = outer.intersection(wedge)
        measurement_areas.append(MeasurementArea(measurement_area_poly))
    
    return walkable_area, measurement_areas


def compute_voronoi_states(traj, walkable_area, measurement_areas, return_polygons=False):
    try:
        individual = compute_individual_voronoi_polygons(
            traj_data=traj, walkable_area=walkable_area,
            cut_off=Cutoff(radius=(1.8+1.5)/2, quad_segments=1)
        )
    except IndexError:      
        plt.figure()
        plot_measurement_setup(
            walkable_area=walkable_area, traj=traj, traj_alpha=0.5, traj_width=1,
            measurement_areas=measurement_areas, ma_line_width=2, ma_alpha=0.5,
        ).set_aspect("equal")
        plt.show()
        sys.exit(1)
    
    individual_speed_single_sided = compute_individual_speed(
        traj_data=traj, frame_step=int(traj.frame_rate), compute_velocity=False,
        speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED,
    )
    individual_joined = individual_speed_single_sided.merge(individual, on=['id', 'frame'], how='inner')
    individual_joined['flow'] = individual_joined['density'] * individual_joined['speed'] # bic/m^2 * m/s = bic/s/m
    # switch flow and speed columns
    individual_joined['temp'] = individual_joined['flow']
    individual_joined['flow'] = individual_joined['speed']
    individual_joined['speed'] = individual_joined['temp']
    individual_joined = individual_joined.drop(columns=['temp'])
    
    voronoi_density_areas, voronoi_speed_areas = [], []
    for ma in measurement_areas:
        density_voronoi, intersecting = compute_voronoi_density(
            individual_voronoi_data=individual, measurement_area=ma
        )
        voronoi_density_areas.append(density_voronoi)
    
        voronoi_speed = compute_voronoi_speed(
            traj_data=traj, individual_voronoi_intersection=intersecting,
            individual_speed=individual_joined, measurement_area=ma,
        )
        voronoi_speed_areas.append(voronoi_speed)
    
    voronoi_states_areas = []
    for i in range(len(measurement_areas)):
        voronoi_states = voronoi_density_areas[i].join(voronoi_speed_areas[i], how="inner")
        voronoi_states = voronoi_states.rename(columns={'speed': 'flow'}) # switch back
        voronoi_states['density'] = voronoi_states['density']*1000.0 # bic/km/m
        voronoi_states['flow'] = voronoi_states['flow']*3600.0 # bic/h/m
        voronoi_states['speed'] = voronoi_states['flow'] / voronoi_states['density'] # bic/h/m / bic/km/m = km/h
        voronoi_states_areas.append(voronoi_states)
    
    voronoi_states_all = pd.concat(voronoi_states_areas)
    voronoi_states_all = voronoi_states_all.rename(columns={
        'density': 'Density', 'flow': 'Flow', 'speed': 'Speed'
    })
    if return_polygons:
        return voronoi_states_all, voronoi_states_areas, individual_joined
    return voronoi_states_all