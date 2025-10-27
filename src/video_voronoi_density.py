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
import cv2
import random
import pathlib

import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# plt.ioff()

from tqdm import tqdm
from pedpy import load_trajectory
from shapely.geometry import Point

from _log_config import create_log_file
create_log_file(logfile = "../logs/CRB_FD_VideoGeneration.log")
from _log_config import enable_logging_overwrite
enable_logging_overwrite()

from _constants import CRB_Config
CRB_Config = CRB_Config()
from _constants import default_drawing_settings as drawing_settings

from tools_video import getNumberOfFramesFromVideo
from tools_video import loadHomography, getFrameHomography
from tools_video import extractFrameFromVideo
from tools_video import transformPointFrom_CARTESIAN_2_PIX
from tools_voronoi import prepare_data_pedpy, define_measurement_setup
from tools_voronoi import compute_voronoi_states
from tools_bfd import aggregate_fd

# #############################################################################
# CONSTANTS
# #############################################################################
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

RELEVANT_VIDEO = 'DJI_20240906103036_0003_D.MP4'
RELEVANT_PART = "PART_2X"
RELEVANT_FRAME_FROM = 2425 + 25 # Video 3, Part 2X
RELEVANT_FRAME_TO = 3720 - 25 # Video 3, Part 2X

# RELEVANT_VIDEO = 'DJI_20240906110027_0011_D.MP4'
# RELEVANT_PART = "PART_2345X"
# RELEVANT_FRAME_FROM = 2275 + 25 # Video 11, Part 2345X
# RELEVANT_FRAME_TO = 6120 - 25 # Video 11, Part 2345X

video_file_path = CRB_Config.video_path + RELEVANT_VIDEO

RELEVANT_FRAME = 2600 # from 5715 till 5843

NUM_SECTORS = 5
MIN_OBS = 50

# #############################################################################
# METHODS
# #############################################################################
def make_faded_circular_crop_frame(frame, center_cart, radius_cart, frame_homography, 
                                   fade_alpha=0.5, blur_radius=5, 
                                   background_color=(255, 255, 255)):
    # pixel coordinates of center of track
    x_pix, y_pix, r_pix = frame_homography
    center_pix = (x_pix, y_pix)

    # edge in cartesian coordinates
    tol = 1.0
    edge_cart = [center_cart[0] + r_c + tol, center_cart[0]]
    edge_pix = transformPointFrom_CARTESIAN_2_PIX(edge_cart, frame_homography)
    radius_pix = np.hypot(edge_pix[0]-center_pix[0], edge_pix[1]-center_pix[1])

    # original frame size
    h, w = frame.shape[:2]

    # Convert to RGB and fade toward white (inside visible region)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    faded_rgb = frame_rgb * fade_alpha + (1 - fade_alpha) * 1.0  # fade to white
    faded_rgb = np.clip(faded_rgb, 0, 1)

    # Build circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    center_int = (int(round(center_pix[0])), int(round(center_pix[1])))
    radius_int = int(round(radius_pix))
    cv2.circle(mask, center_int, radius_int, 255, -1, cv2.LINE_AA)

    # Optional blur for soft edge
    if blur_radius > 0:
        k = int(blur_radius) * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), blur_radius)

    # Crop bounding box around circle
    x0 = max(0, center_int[0] - radius_int)
    y0 = max(0, center_int[1] - radius_int)
    x1 = min(w, center_int[0] + radius_int)
    y1 = min(h, center_int[1] + radius_int)
    crop_rgb = (faded_rgb[y0:y1, x0:x1] * 255).astype(np.uint8)
    crop_mask = mask[y0:y1, x0:x1]

    # Make RGBA (alpha = mask)
    rgba = np.dstack([crop_rgb, crop_mask])

    return rgba, (x0, y0, x1 - x0, y1 - y0)


def draw_voronoi_polygons_on_frame(rgba_frame, polygons_df, bbox, bicycle_colors=None,
                                   alpha=0.15, font=cv2.FONT_HERSHEY_SIMPLEX, 
                                   font_scale=2.25, font_thickness=3, transparent_polygons=True):
    # border_color=(0, 0, 0, 255) #(255, 255, 255, 255)
    overlay = rgba_frame.copy().astype(np.uint8)
    x0, y0, w, h = bbox

    for idx, row in polygons_df.iterrows():
        bike_id = f"BICYCLE_{row['id']}"
        poly_cart = np.array(row['polygon'].exterior.coords)
        poly_pix = np.zeros_like(poly_cart)
        for i in range(poly_cart.shape[0]):
            poly_pix[i, :] = transformPointFrom_CARTESIAN_2_PIX(poly_cart[i, :], frame_homography)
        # Adjust to cropped coordinate system
        poly_pix[:, 0] -= x0
        poly_pix[:, 1] -= y0
        poly_pix = np.round(poly_pix).astype(np.int32)
        # Draw filled polygon (on an intermediate layer)
        mask = np.zeros_like(overlay, dtype=np.uint8)
        
        if transparent_polygons:
            border_color = (0, 0, 0, 255)
        else:
            # Pick color for this bike
            color_rgb = bicycle_colors.get(bike_id, (0, 255, 0))
            color_rgb = tuple(int(255 * c) for c in mcolors.to_rgb(color_rgb))
            fill_color = (*color_rgb, int(alpha * 255))  # add alpha
            border_color = (*color_rgb, 255)
            cv2.fillPoly(mask, [poly_pix], color=fill_color)
            
        cv2.polylines(mask, [poly_pix], isClosed=True, color=border_color, thickness=2)
        
        # Alpha blend the mask with the overlay
        alpha_mask = mask[:, :, 3] / 255.0
        for c in range(3):
            overlay[:, :, c] = (1 - alpha_mask) * overlay[:, :, c] + alpha_mask * mask[:, :, c]
        
        poly_top = np.min(poly_pix[:, 1])   # smallest y value = topmost point in image coords
        poly_xmin, poly_xmax = np.min(poly_pix[:, 0]), np.max(poly_pix[:, 0])
        cx = (poly_xmin + poly_xmax) / 2    # horizontal center
        cy = poly_top - 10                  # 10 px above the top edge (tweak as needed)
        text = f"B_{row['id']}"
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.putText(
            overlay,
            text,
            (int(cx) - (text_width // 2) - 10, int(cy) + (text_height // 2)  - 20),
            font,
            font_scale,
            (0, 0, 0, 255),  # opaque black
            font_thickness,
            cv2.LINE_AA
        )
    return overlay


def draw_colorcoded_measurement_areas(rgba_frame, measurement_areas, frame_voronoi_states,
                                      r_inner, bbox, state='density', vmin=0, 
                                      vmax=120, cmap_name='RdYlGn_r', alpha=0.5, gamma=0,
                                      border_color=(0, 0, 0, 255), border_thickness=2):
    overlay = np.zeros_like(rgba_frame, dtype=np.uint8)
    x0, y0, _, _ = bbox
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.get_cmap(cmap_name)

    inner_circle = Point(x_c, y_c).buffer(r_inner, quad_segs=64)
    for i in range(len(measurement_areas)):
        ring_part = measurement_areas[i].polygon.difference(inner_circle)
        poly_cart = np.array(ring_part.exterior.coords)
        poly_pix = np.zeros_like(poly_cart)
        for j in range(poly_cart.shape[0]):
            poly_pix[j, :] = transformPointFrom_CARTESIAN_2_PIX(poly_cart[j, :], frame_homography)
        # Adjust to cropped coordinate system
        poly_pix[:, 0] -= x0
        poly_pix[:, 1] -= y0
        poly_pix = np.round(poly_pix).astype(np.int32)
        
        # Map Traffic State → RGB color
        ts = frame_voronoi_states[i][state]
        color = np.array(cmap(norm(ts))[:3]) * 255
        fill_color = tuple(np.append(color.astype(np.uint8), 255).tolist()) # int(alpha * 255)
        # border_color = tuple(np.append(color.astype(np.uint8), 255).tolist())
        
        cv2.fillPoly(overlay, [poly_pix], color=fill_color)
        cv2.polylines(overlay, [poly_pix], isClosed=True, color=border_color, thickness=border_thickness)
        
    overlay = cv2.addWeighted(rgba_frame, 1 - alpha, overlay, alpha, gamma)
    return overlay


def add_colorbar_right(rgba_frame, cmap_name='RdYlGn', vmin=0, vmax=120, alpha=0.5,
                       label='Density [bic/km/m]', n_ticks=5, tick_format=".0f", 
                       width=60, padding=100, font_scale=2.5, font_thickness=2,
                       tick_color=(0,0,0), tick_length=25, tick_thickness=4,
                       tick_label_offset=10, bar_margin_ratio=0.05):
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.get_cmap(cmap_name)

    h, w = rgba_frame.shape[:2]
    has_alpha = (rgba_frame.shape[2] == 4)

    # New extended frame (same height, wider)
    new_w = w + width + 5*padding
    extended = np.ones((h, new_w, rgba_frame.shape[-1]), dtype=np.uint8) * 255  # white background

    # Place the original frame on the left
    extended[:, :w] = rgba_frame

    # Compute reduced colorbar height
    bar_top = int(h * bar_margin_ratio)
    bar_bottom = int(h * (1 - bar_margin_ratio))
    bar_h = bar_bottom - bar_top

    # Prepare colormap vertical gradient: shape (h, 3) in RGB
    gradient = np.linspace(vmax, vmin, bar_h)[:, None]  # top -> bottom
    colors_rgb = (cmap(norm(gradient))[:, 0, :3] * 255).astype(np.uint8)  # (h,3)
    colorbar_rgb = np.repeat(colors_rgb[:, None, :], width, axis=1)
    # colorbar_bgr = colorbar_rgb[:, :, ::-1].copy()  # (h, width, 3)
    # If frame has alpha, make BGRA colorbar and set alpha=255
    if has_alpha:
        alpha_channel = np.full((bar_h, width, 1), 255*alpha, dtype=np.uint8)
        colorbar_img = np.concatenate([colorbar_rgb, alpha_channel], axis=2)  # (h,width,4)
    else:
        colorbar_img = colorbar_rgb  # (h,width,3)

    # Paste colorbar in the new region
    x0 = w + padding
    x1 = x0 + width
    extended[bar_top:bar_bottom, x0:x1] = colorbar_img

    # Add tick labels to the right of the colorbar
    for i, v in enumerate(np.linspace(vmin, vmax, n_ticks)):
        y = int(bar_bottom - i * (bar_h / (n_ticks - 1)))  # fit ticks to the shorter bar
        text = format(v, tick_format)
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_w, text_h = text_size
        
        # --- Draw tick line (filled rectangle instead of cv2.line) ---
        tick_start_x = x1 - 10
        tick_end_x = tick_start_x + tick_length
        tick_half = tick_thickness // 2
        
        y1_tick = max(0, y - tick_half)
        y2_tick = min(extended.shape[0], y + tick_half)
        x1_tick = tick_start_x
        x2_tick = tick_end_x

        if has_alpha:
            # Handle RGBA case
            tick_color_rgba = (*tick_color, int(255 * alpha))  # e.g., (0,0,0,128)
            extended[y1_tick:y2_tick, x1_tick:x2_tick] = tick_color_rgba
        else:
            # Handle RGB case
            extended[y1_tick:y2_tick, x1_tick:x2_tick] = tick_color  
        
        
        # --- Draw filled tick label using mask ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = tick_end_x + 15
        text_y = y + text_h // 2

        # create a small mask for the text
        mask = np.zeros((text_h + 10, text_w + 10), dtype=np.uint8)
        cv2.putText(mask, text, (5, text_h + 5), font, font_scale, 255, font_thickness, cv2.LINE_AA)

        # define ROI on extended image
        y1 = max(0, text_y - text_h)
        y2 = min(extended.shape[0], text_y + 10)
        x1m = max(0, text_x)
        x2m = min(extended.shape[1], text_x + text_w + 10)

        # align mask within ROI
        mask_h, mask_w = mask.shape
        roi = extended[y1:y1 + mask_h, x1m:x1m + mask_w]

        if has_alpha:
            roi[mask > 0] = (*tick_color, 255)
        else:
            roi[mask > 0] = tick_color

        extended[y1:y1 + mask_h, x1m:x1m + mask_w] = roi

    # Prepare vertical (rotated) label using a tight image
    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale*1.25, font_thickness)
    pad = 50
    txt_w = label_w + 2 * pad
    txt_h = label_h + 2 * pad

    # If frame has alpha channel, keep 4 channels; otherwise 3 channels
    channels = rgba_frame.shape[2]
    bg_color = 255
    label_img = np.full((txt_h, txt_w, channels), bg_color, dtype=np.uint8)

    # Put text into the small label image. baseline correction: putText uses baseline at y coordinate.
    mask = np.zeros((txt_h, txt_w), dtype=np.uint8)
    text_org = (pad, pad + label_h)
    cv2.putText(mask, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale*1.25, 255, font_thickness, cv2.LINE_AA)
    if has_alpha:
        label_img[mask > 0] = (*tick_color, 255)  # solid color with full alpha
    else:
        label_img[mask > 0] = tick_color  # solid BGR fill

    # Rotate 90 degrees CCW (so text reads vertically, bottom->top like Matplotlib)
    rotated_label = cv2.rotate(label_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rh, rw = rotated_label.shape[:2]

    # Compute paste position: to the right of tick labels, vertically centered
    paste_x = x1 + tick_length + text_w + tick_label_offset
    paste_y = max(0, (h - rh) // 2)

    # Bounds check (shrink if necessary)
    if paste_x + rw > extended.shape[1]:
        # clamp paste_x so it fits
        paste_x = extended.shape[1] - rw - 1
    if paste_y + rh > extended.shape[0]:
        paste_y = extended.shape[0] - rh - 1

    # Create mask where rotated_label is not white (assuming background is white)
    if has_alpha:
        # check all channels against 255
        mask = ~(np.all(rotated_label[:, :, :3] == 255, axis=2))
    else:
        mask = ~(np.all(rotated_label == 255, axis=2))

    # Paste with mask
    roi = extended[paste_y:paste_y + rh, paste_x:paste_x + rw]
    roi[mask] = rotated_label[mask]
    extended[paste_y:paste_y + rh, paste_x:paste_x + rw] = roi
    # extended now contains the rotated vertical label centered beside the colorbar
    return extended


def plot_BFD_operating_point(fd_fig, current_density, highlight_color='blue', 
                             fs_ticks=18, fs_labels=22):
    axs = fd_fig.get_axes()
    newfig, newaxs = plt.subplots(2, 1, figsize=(5, 8), sharex=True, dpi=300)

    line_free_pfd = axs[0].lines[0].get_xydata()
    line_cong_pfd = axs[0].lines[1].get_xydata()
    scatter_pfd = axs[0].collections[0].get_offsets()
    newaxs[0].scatter(scatter_pfd[:, 0], scatter_pfd[:, 1], color='black', alpha=0.1)
    newaxs[0].plot(line_free_pfd[:, 0], line_free_pfd[:, 1], color="green", linestyle="dashed", linewidth=2)
    newaxs[0].plot(line_cong_pfd[:, 0], line_cong_pfd[:, 1], color="red", linestyle="dashed", linewidth=2)
    newaxs[0].axvspan(current_density+0.3, current_density-0.3, color=highlight_color, alpha=0.25)
    newaxs[0].set(xlim=[0, 160], ylim=[0, 1500])
    newaxs[0].set_ylabel('Flow [bic/h/m]', fontsize=fs_labels)
    newaxs[0].tick_params(axis='both', labelsize=fs_ticks)
    newaxs[0].locator_params(axis='x', nbins=5, tight=True)
    newaxs[0].locator_params(axis='y', nbins=5, tight=True)

    line_free_pfd = axs[1].lines[0].get_xydata()
    line_cong_pfd = axs[1].lines[1].get_xydata()
    scatter_pfd = axs[1].collections[0].get_offsets()
    newaxs[1].scatter(scatter_pfd[:, 0], scatter_pfd[:, 1], color='black', alpha=0.1)
    newaxs[1].plot(line_free_pfd[:, 0], line_free_pfd[:, 1], color="green", linestyle="dashed", linewidth=2)
    newaxs[1].plot(line_cong_pfd[:, 0], line_cong_pfd[:, 1], color="red", linestyle="dashed", linewidth=2)
    newaxs[1].axvspan(current_density+0.3, current_density-0.3, color=highlight_color, alpha=0.25)
    newaxs[1].set(xlim=[0, 160], ylim=[0, 20])
    newaxs[1].set_xlabel('Density [bic/km/m]', fontsize=fs_labels)
    newaxs[1].set_ylabel('Speed [km/h]', fontsize=fs_labels)
    newaxs[1].tick_params(axis='both', labelsize=fs_ticks)
    newaxs[1].locator_params(axis='x', nbins=5, tight=True)
    newaxs[1].locator_params(axis='y', nbins=5, tight=True)
    
    newfig.tight_layout()
    
    # Convert figure to RGBA image
    newfig.canvas.draw()
    img_rgba = np.frombuffer(newfig.canvas.buffer_rgba(), dtype=np.uint8)
    img_rgba = img_rgba.reshape(newfig.canvas.get_width_height()[::-1] + (4,))
    plt.close(newfig)
    return img_rgba


def add_BFD_to_right(overlay, fd_rgba, padding=80, bg_color=255):
    """
    Append the fundamental diagram (fd_rgba) to the right of an overlay frame.
    Handles size mismatch and alpha blending.
    """
    h, w = overlay.shape[:2]
    has_alpha = (overlay.shape[2] == 4)

    fd_h, fd_w = fd_rgba.shape[:2]
    scale = h / fd_h * 0.95  # make it slightly smaller vertically
    fd_resized = cv2.resize(fd_rgba, (int(fd_w * scale), int(fd_h * scale)), interpolation=cv2.INTER_AREA)
    fd_h, fd_w = fd_resized.shape[:2]

    # New extended frame
    new_w = w + fd_w + padding
    extended = np.ones((h, new_w, overlay.shape[2]), dtype=np.uint8) * bg_color

    # Paste overlay
    extended[:, :w] = overlay

    # Paste the fundamental diagram centered vertically
    y0 = (h - fd_h) // 2
    x0 = w + padding
    x1 = x0 + fd_w

    if has_alpha:
        alpha_fd = fd_resized[:, :, 3:] / 255.0
        alpha_bg = 1.0 - alpha_fd
        extended[y0:y0+fd_h, x0:x1, :3] = (
            alpha_fd * fd_resized[:, :, :3] +
            alpha_bg * extended[y0:y0+fd_h, x0:x1, :3]
        ).astype(np.uint8)
    else:
        extended[y0:y0+fd_h, x0:x1, :3] = fd_resized[:, :, :3]

    return extended
    

# #############################################################################
# LOADING - FILES
# #############################################################################
# VIDEO
num_frames =  getNumberOfFramesFromVideo(video_file_path)
# HOMOGRAPHY
df_homography = loadHomography(CRB_Config.homography_root+RELEVANT_VIDEO+"_homography3.txt")
# FINAL TRAJECTORY
df_final_trajectory = pd.read_csv(CRB_Config.data_root+RELEVANT_VIDEO+"_"+RELEVANT_PART+".txt")
df_final_trajectory['Vehicle_ID_Sep'] = df_final_trajectory['Vehicle_ID'].str.split('_')
df_final_trajectory['id'] = df_final_trajectory['Vehicle_ID_Sep'].apply(lambda x: x[1])
df_final_trajectory = df_final_trajectory.drop(columns=['Vehicle_ID_Sep'])


# VORONOI
filepath = f"../data/pedpy_traj/{RELEVANT_VIDEO}_{RELEVANT_PART}.txt"
if not os.path.exists(filepath):
    df = pd.read_csv(CRB_Config.data_root + f"{RELEVANT_VIDEO}_{RELEVANT_PART}.txt")
    df = prepare_data_pedpy(df, CRB_Config.video_lane_widths[RELEVANT_VIDEO], config=CRB_Config)
    df.to_csv(filepath, index=False, sep='\t', header=False)
    del df
traj = load_trajectory(
    trajectory_file=pathlib.Path(filepath),
    default_frame_rate=25,
    default_unit='m'
)
walkable_area, measurement_areas = define_measurement_setup(NUM_SECTORS, CRB_Config.video_lane_widths[RELEVANT_VIDEO])
_, voronoi_states_areas, individual_joined = compute_voronoi_states(traj, walkable_area, measurement_areas, return_polygons=True)
del traj
gc.collect()

# Fundamental Diagram
# LANE WIDTH 2.5 METERS
video_set = CRB_Config.videos[:-3]
if RELEVANT_VIDEO in video_set:
    max_density = 150
else:
    # LANE WIDTH 3.75 METERS
    video_set = CRB_Config.videos[-3:]
    max_density = 110
pfd_df_all_X = pd.read_csv("../data/CRB_PseudoTrafficStates_ALLVideos_V2.txt")

tmp_df = pfd_df_all_X[pfd_df_all_X['Video'].isin(video_set)].copy()
tmp_df['v_Vel'] = tmp_df['Space_Hdwy'] / tmp_df['Time_Hdwy']
tmp_df = tmp_df[(tmp_df['v_Vel'] - 0.1).abs() <= 1e-02]
jam_density_est = 1000/tmp_df['Space_Hdwy'].median() / CRB_Config.video_lane_widths[RELEVANT_VIDEO]
_, fd_fig = aggregate_fd(pfd_df_all_X[pfd_df_all_X['Video'].isin(video_set)], 
                      max_density=200, bin_width=0.3, min_observations=MIN_OBS, 
                      FD_form="WuFD", loss_fn="HuberLoss", jam_density=jam_density_est,
                      show_pseudo_states=False, log_results=False)
plt.close(fd_fig)
del pfd_df_all_X, tmp_df
gc.collect()

# #############################################################################
# MAIN: SINGLE FRAME
# #############################################################################
# FRAME, HOMOGRAPHY
success, frame = extractFrameFromVideo(video_file_path, RELEVANT_FRAME)
frame_homography = getFrameHomography(df_homography, RELEVANT_FRAME)
frame_trajectory = df_final_trajectory[df_final_trajectory['Frame_ID'] == RELEVANT_FRAME].copy()

frame_individual_joined = individual_joined[individual_joined['frame'] == RELEVANT_FRAME].copy()
frame_voronoi_states = []
for i in range(len(measurement_areas)):
    frame_voronoi_states.append(voronoi_states_areas[i].loc[RELEVANT_FRAME].to_dict())

# cartesian coordinates of center of track
x_c, y_c, r_c = 0, 0, CRB_Config.circle_outer_radius 
center_cart = [x_c, y_c]

rgba_crop, bbox = make_faded_circular_crop_frame(
    frame, center_cart, r_c, frame_homography, fade_alpha=0.5, blur_radius=5
)
plt.figure(figsize=(5,5))
plt.imshow(rgba_crop)
plt.axis('off')
plt.title("Faded circular crop (RGBA)")
plt.show()

overlay_frame = draw_voronoi_polygons_on_frame(
    rgba_crop, frame_individual_joined, bbox, drawing_settings['bicycle_colors'], 
    alpha=0.15, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=2.25, font_thickness=3,
    transparent_polygons=True
)
    
plt.figure(figsize=(5,5))
plt.imshow(overlay_frame)
plt.axis('off')
plt.title("Voronoi Polygons Overlayed on Faded Frame")
plt.show()

r_inner = CRB_Config.circle_outer_radius - CRB_Config.video_lane_widths[RELEVANT_VIDEO]
overlay = draw_colorcoded_measurement_areas(
    overlay_frame, measurement_areas, frame_voronoi_states, r_inner, bbox, 
    state='speed', vmin=0, vmax=20, cmap_name='RdYlGn', alpha=0.25, gamma=10,
    border_color=(0, 0, 0, 255), border_thickness=2
)
plt.figure(figsize=(5,5))
plt.imshow(overlay)
plt.axis('off')
plt.title("Measurement Areas Overlayed")
plt.show()


# Add colorbar
cbar_overlay = add_colorbar_right(
    overlay, cmap_name='RdYlGn', vmin=0, vmax=20, alpha=0.5, label='Speed [km/h]', 
    n_ticks=5, tick_format=".0f", width=60, padding=100, font_scale=2.5, font_thickness=2,
    tick_color=(0,0,0), tick_length=25, tick_thickness=4, tick_label_offset=10, 
    bar_margin_ratio=0.05
)

plt.figure(figsize=(5,5))
plt.imshow(cbar_overlay)
plt.axis('on')
plt.title("Colorbar Overlayed")
plt.show()

current_density = 0
weights = 0
mean_density = 0
for i in range(len(measurement_areas)):
    w = frame_voronoi_states[i]['density'] / max_density
    current_density += w*frame_voronoi_states[i]['density']
    mean_density += frame_voronoi_states[i]['density']
    weights += w
current_density /= weights
mean_density /= len(measurement_areas)
fd_img_rgba = plot_BFD_operating_point(fd_fig, current_density, highlight_color='blue')
overlay_fd = add_BFD_to_right(cbar_overlay, fd_img_rgba, padding=150, bg_color=255)

# Convert RGBA → BGR (with white background)
if overlay_fd.shape[2] == 4:
    alpha = overlay_fd[:, :, 3:] / 255.0
    rgb = overlay_fd[:, :, :3]
    white_bg = np.ones_like(rgb, dtype=np.uint8) * 255
    bgr_frame = cv2.cvtColor((rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8), cv2.COLOR_RGB2BGR)
else:
    bgr_frame = cv2.cvtColor(overlay_fd, cv2.COLOR_RGB2BGR)

plt.figure()
plt.imshow(overlay_fd)
plt.axis('off')

cv2.namedWindow('Cropped Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Cropped Image', 800, 600)
cv2.imshow('Cropped Image', bgr_frame)

sys.exit(1)


# #############################################################################
# MAIN: GENERATING VIDEO
# #############################################################################
output_video_path = os.path.join(CRB_Config.video_path, RELEVANT_VIDEO.split('.')[0]+"_"+RELEVANT_PART+"_Voronoi.mp4")

temp_dir = "./temp_frames"
os.makedirs(temp_dir, exist_ok=True)

fps = CRB_Config.sampling_freq  # set your video frame rate
frame_size = None  # will infer after first frame

sys.exit(1)
# --- MAIN LOOP ---
for frame_nr in tqdm(range(RELEVANT_FRAME_FROM, RELEVANT_FRAME_TO+1), total=RELEVANT_FRAME_TO-RELEVANT_FRAME_FROM+1, desc="Rendering video"):
    # Step 1: Load Frame Annotations
    success, frame = extractFrameFromVideo(video_file_path, frame_nr)
    frame_homography = getFrameHomography(df_homography, frame_nr)
    frame_trajectory = df_final_trajectory[df_final_trajectory['Frame_ID'] == frame_nr].copy()

    frame_individual_joined = individual_joined[individual_joined['frame'] == frame_nr].copy()
    frame_voronoi_states = []
    for i in range(len(measurement_areas)):
        frame_voronoi_states.append(voronoi_states_areas[i].loc[frame_nr].to_dict())
    
    # Step 2: Faded crop
    x_c, y_c, r_c = 0, 0, CRB_Config.circle_outer_radius
    center_cart = [x_c, y_c]
    rgba_crop, bbox = make_faded_circular_crop_frame(
        frame, center_cart, r_c, frame_homography, fade_alpha=0.5, blur_radius=5
    )

    # Step 3: Voronoi polygons overlay
    overlay_frame = draw_voronoi_polygons_on_frame(
        rgba_crop, frame_individual_joined, bbox, drawing_settings['bicycle_colors'],
        alpha=0.15, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=2.25, font_thickness=3,
        transparent_polygons=True
    )

    # Step 4: Measurement areas overlay
    r_inner = CRB_Config.circle_outer_radius - CRB_Config.video_lane_widths[RELEVANT_VIDEO]
    overlay = draw_colorcoded_measurement_areas(
        overlay_frame, measurement_areas, frame_voronoi_states, r_inner, bbox,
        state='speed', vmin=0, vmax=20, cmap_name='RdYlGn', alpha=0.25, gamma=10,
        border_color=(0, 0, 0, 255), border_thickness=2
    )

    # Step 5: Add colorbar to the right
    cbar_overlay = add_colorbar_right(
        overlay, cmap_name='RdYlGn', vmin=0, vmax=20, alpha=0.5,
        label='Speed [km/h]', n_ticks=5, tick_format=".0f",
        width=60, padding=100, font_scale=2.5, font_thickness=2,
        tick_color=(0, 0, 0), tick_length=25, tick_thickness=4,
        tick_label_offset=10, bar_margin_ratio=0.05
    )
    
    # Step 6: Add BFD to the right
    current_density = 0
    weights = 0
    for i in range(len(measurement_areas)):
        w = frame_voronoi_states[i]['density'] / max_density
        current_density += w*frame_voronoi_states[i]['density']
        weights += w
    current_density /= weights
    fd_img_rgba = plot_BFD_operating_point(fd_fig, current_density, highlight_color='blue')
    overlay_fd = add_BFD_to_right(cbar_overlay, fd_img_rgba, padding=150, bg_color=255)
   
    # Step 7: Convert RGBA → BGR (with white background)
    if overlay_fd.shape[2] == 4:
        alpha = overlay_fd[:, :, 3:] / 255.0
        rgb = overlay_fd[:, :, :3]
        white_bg = np.ones_like(rgb, dtype=np.uint8) * 255
        bgr_frame = cv2.cvtColor((rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        bgr_frame = cv2.cvtColor(overlay_fd, cv2.COLOR_RGB2BGR)

    # Step 8: Save temporary frame (optional, helps memory)
    temp_path = os.path.join(temp_dir, f"frame_{frame_nr:05d}.png")
    cv2.imwrite(temp_path, bgr_frame)

    if frame_size is None:
        frame_size = (bgr_frame.shape[1], bgr_frame.shape[0])
        
    gc.collect()


# --- CREATE VIDEO FROM SAVED FRAMES ---
frame_size = (4160, 2157)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

if not writer.isOpened():
    print("❌ VideoWriter failed to open!")

frame_files = sorted(os.listdir(temp_dir))
print("Found", len(frame_files), "frames.")
for fname in tqdm(frame_files, desc="Encoding video"):
    frame = cv2.imread(os.path.join(temp_dir, fname))
    if frame is None:
        print("⚠️ Skipping unreadable frame:", fname)
        sys.exit(1)
    if (frame.shape[1], frame.shape[0]) != frame_size:
        # print("⚠️ Resizing:", fname)
        frame = cv2.resize(frame, frame_size)
    writer.write(frame)

writer.release()
cv2.destroyAllWindows()
print(f"\n✅ Video saved to: {output_video_path}")

# # --- OPTIONAL: CLEAN UP TEMP FILES ---
# import shutil
# shutil.rmtree(temp_dir)
# print("🧹 Temporary frames deleted.")