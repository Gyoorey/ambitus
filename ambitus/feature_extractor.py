import os
import csv
import json
import numpy as np
import click
import cv2
from itertools import groupby

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# GEOMETRIC HELPERS
# ==============================================================================
def get_polygon_centroid(poly):
    M = cv2.moments(poly)
    if M["m00"] != 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return (0, 0)

def calculate_angle(p1, p2):
    return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

def normalize_angle_diff(diff):
    while diff <= -180: diff += 360
    while diff > 180: diff -= 360
    return diff

def get_sector(x, y, cx, cy):
    angle = np.degrees(np.arctan2(y - cy, x - cx))
    if angle < 0: angle += 360  
    shifted = (angle + 22.5) % 360
    sector_id = int(shifted // 45)
    is_corner = (sector_id % 2 != 0)
    return sector_id, is_corner

# ==============================================================================
# DEBUG VISUALIZATION (PLOTLY)
# ==============================================================================
def generate_feature_dashboard(times, speeds, states, t_lower, t_upper, 
                               xs, ys, arena_center, directions, 
                               minute_distances, html_path):
    """
    Creates a 3-panel HTML dashboard to debug the feature extraction logic.
    """
    # Create subplots with explicitly INDEPENDENT axes
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=(
            "1. Kinematics & States (Speed vs Time)", 
            "2. Spatial & Directional (CW/CCW map)", 
            "3. Habituation (Distance per Minute)"
        ),
        row_heights=[0.5, 0.5],
        shared_xaxes=False,  # DO NOT link the X axes!
        shared_yaxes=False   # DO NOT link the Y axes!
    )

    # --- PLOT 1: SPEED & STATES (Top spanning row) ---
    fig.add_trace(go.Scatter(
        x=times[:-1], y=speeds, mode='lines', line=dict(color='black', width=1), name='Speed (px/s)'
    ), row=1, col=1)

    fig.add_hline(y=t_upper, line_dash="dash", line_color="green", annotation_text="T_upper", row=1, col=1)
    fig.add_hline(y=t_lower, line_dash="dash", line_color="red", annotation_text="T_lower", row=1, col=1)

    # Variables to ensure we only add the legend for the background ONCE
    active_legend_added = False
    inactive_legend_added = False

    start_idx = 0
    for key, group in groupby(states):
        length = sum(1 for _ in group)
        end_idx = min(start_idx + length, len(times) - 2)
        
        t0 = times[start_idx]
        t1 = times[end_idx]
        
        if key == 1:
            color = "rgba(0, 255, 0, 0.2)"
            name = "Active Phase"
            show_leg = not active_legend_added
            active_legend_added = True
        else:
            color = "rgba(255, 0, 0, 0.2)"
            name = "Inactive Phase"
            show_leg = not inactive_legend_added
            inactive_legend_added = True
            
        fig.add_vrect(
            x0=t0, x1=t1, fillcolor=color, layer="below", line_width=0, 
            row=1, col=1
        )
        
        # Add a dummy invisible scatter trace just to force the legend to appear for the vrects
        if show_leg:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=10, color=color.replace('0.2', '0.8'), symbol='square'),
                name=name
            ), row=1, col=1)
            
        start_idx += length

    # --- PLOT 2: SPATIAL MAP & DIRECTIONS (Bottom Left) ---
    cw_x, cw_y = [], []
    ccw_x, ccw_y = [], []
    
    for i in range(len(directions)):
        if directions[i] == 1:
            cw_x.append(xs[i]); cw_y.append(ys[i])
        elif directions[i] == -1:
            ccw_x.append(xs[i]); ccw_y.append(ys[i])
            
    fig.add_trace(go.Scatter(x=cw_x, y=cw_y, mode='markers', marker=dict(color='blue', size=3), name='CW Move'), row=2, col=1)
    fig.add_trace(go.Scatter(x=ccw_x, y=ccw_y, mode='markers', marker=dict(color='orange', size=3), name='CCW Move'), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=[arena_center[0]], y=[arena_center[1]], mode='markers', marker=dict(color='black', symbol='cross', size=10), name='Arena Center'), row=2, col=1)

    # Ensure the spatial map looks like a map (inverted Y, 1:1 aspect ratio)
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, row=2, col=1)

    # --- PLOT 3: HABITUATION BAR CHART (Bottom Right) ---
    mins = [f"Min {i+1}" for i in range(len(minute_distances))]
    fig.add_trace(go.Bar(x=mins, y=minute_distances, marker_color='purple', name='Distance/Min'), row=2, col=2)

    # --- LAYOUT & AXES SETTINGS ---
    # Force the Top plot to start at X=0 and end at the last valid timestamp + 5s margin
    max_time = times[-1] if len(times) > 0 else 300
    fig.update_xaxes(range=[0, max_time + 5], row=1, col=1)
    
    fig.update_xaxes(title_text="Time (sec)", row=1, col=1)
    fig.update_yaxes(title_text="Speed (px/s)", row=1, col=1)
    
    fig.update_xaxes(title_text="X Pixel", row=2, col=1)
    fig.update_yaxes(title_text="Y Pixel", row=2, col=1)
    
    fig.update_yaxes(title_text="Distance (px)", row=2, col=2)

    fig.update_layout(
        height=800, 
        width=1200, 
        title_text="Feature Extraction Debug Dashboard", 
        showlegend=True, # Legend is now ON
        hovermode="closest"
    )
    
    fig.write_html(html_path)
    print(f"Debug Dashboard saved to: {html_path}")

# ==============================================================================
# FEATURE EXTRACTION ENGINE
# ==============================================================================
def extract_features(csv_path, config_path, output_json_path, debug_html=False, global_thresholds_path=None):
    if not os.path.exists(csv_path) or not os.path.exists(config_path):
        print(f"ERROR: Missing input files for extraction.")
        return

    # --- 1. LOAD CONFIG ---
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    inner_poly = np.array(config_data["inner_poly"], dtype=np.int32)
    arena_center = get_polygon_centroid(inner_poly)

    # --- 2. LOAD DATA ---
    times, xs, ys = [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader, None)
        for row in reader:
            try:
                times.append(float(row[0]))
                xs.append(float(row[1]))
                ys.append(float(row[2]))
            except (ValueError, IndexError):
                pass
    if len(times) < 10: return

    times, xs, ys = np.array(times), np.array(xs), np.array(ys)
    total_time = times[-1] - times[0]

    # --- 3. KINEMATICS ---
    dt = np.diff(times)  
    dx = np.diff(xs)
    dy = np.diff(ys)
    distances = np.sqrt(dx**2 + dy**2)
    speeds = distances / dt
    
    accelerations = np.diff(speeds) / dt[:-1]
    accel_variance = np.var(accelerations) if len(accelerations) > 0 else 0.0

    total_distance = np.sum(distances)

    # --- 4. ADAPTIVE THRESHOLDS ---
    t_lower, t_upper = 2.0, 5.0 # Fallback default
    used_threshold_type = "Local (Per-Video)"

    avg_dt = total_time / len(times) if len(times) > 0 else 0.04
    noise_floor_speed = 1.0 / avg_dt if avg_dt > 0 else 25.0

    if global_thresholds_path and os.path.exists(global_thresholds_path):
        with open(global_thresholds_path, 'r') as f:
            glob_data = json.load(f)
            if "T_lower_px_s" in glob_data and "T_upper_px_s" in glob_data:
                t_lower = glob_data["T_lower_px_s"]
                t_upper = glob_data["T_upper_px_s"]
                used_threshold_type = "Global (Population)"
    else:
        # If no global file provided, calculate locally based on this specific video
        moving_speeds = speeds[speeds > noise_floor_speed]
        if len(moving_speeds) > 10:
            t_lower = np.percentile(moving_speeds, 15)
            t_upper = np.percentile(moving_speeds, 40)

    # Ensure thresholds are reasonable (not below noise floor, and upper > lower)
    if t_lower < noise_floor_speed:
        t_lower = noise_floor_speed
    if t_upper <= t_lower:
        t_upper = t_lower * 1.5

    # --- 5. HYSTERESIS & DEBOUNCING ---
    states = np.zeros(len(speeds), dtype=int)
    current_state = 0
    for i, v in enumerate(speeds):
        if v > t_upper: current_state = 1
        elif v < t_lower: current_state = 0
        states[i] = current_state

    # Temporal Smoothing (Debounce)
    fps = 1.0 / np.median(dt) if len(dt) > 0 else 25.0
    min_gap_frames = int(1.0 * fps) 
    min_spike_frames = int(0.5 * fps) 
    
    # Fill short gaps (0s inside 1s)
    start_idx = 0
    for key, group in groupby(states):
        length = sum(1 for _ in group)
        if key == 0 and length <= min_gap_frames:
            states[start_idx : start_idx + length] = 1
        start_idx += length

    # Remove short spikes (1s inside 0s)
    start_idx = 0
    for key, group in groupby(states):
        length = sum(1 for _ in group)
        if key == 1 and length <= min_spike_frames:
            states[start_idx : start_idx + length] = 0
        start_idx += length

    # --- Group Phases ---
    active_phases, inactive_phases = [], []
    start_idx = 0
    for key, group in groupby(states):
        length = sum(1 for _ in group)
        end_idx = min(start_idx + length, len(times) - 1)
        phase_duration = times[end_idx] - times[start_idx]
        
        if key == 1:
            phase_dist = np.sum(distances[start_idx:end_idx])
            active_phases.append({'duration': phase_duration, 'distance': phase_dist})
        else:
            inactive_phases.append({'duration': phase_duration})
        start_idx += length

    short_active = [p for p in active_phases if p['duration'] <= 5.0]
    long_active = [p for p in active_phases if p['duration'] > 5.0]
    short_inactive = [p for p in inactive_phases if p['duration'] <= 5.0]
    long_inactive = [p for p in inactive_phases if p['duration'] > 5.0]

    def safe_mean(lst, key=None):
        if not lst: return 0.0
        if key: return float(np.mean([item[key] for item in lst]))
        return float(np.mean(lst))
        
    def safe_sum(lst, key=None):
        if not lst: return 0.0
        if key: return float(np.sum([item[key] for item in lst]))
        return float(np.sum(lst))

    # --- 6. SPATIAL & DIRECTIONAL ANALYSIS ---
    corner_time = 0.0
    cw_dist, ccw_dist = 0.0, 0.0
    u_turns = 0
    
    angles_to_center = [calculate_angle(arena_center, (x, y)) for x, y in zip(xs, ys)]
    sector_counts = np.zeros(8, dtype=int)
    current_direction = 0 
    
    # Store directions for debug plot (1 = CW, -1 = CCW, 0 = Unknown)
    directions_plot = np.zeros(len(xs)-1, dtype=int)
    
    for i in range(len(xs) - 1):
        sec_id, is_corner = get_sector(xs[i], ys[i], arena_center[0], arena_center[1])
        sector_counts[sec_id] += 1
        
        if is_corner: corner_time += dt[i]

        angle_diff = normalize_angle_diff(angles_to_center[i+1] - angles_to_center[i])
        
        if angle_diff > 0.5: # CW
            cw_dist += distances[i]
            directions_plot[i] = 1
            if current_direction == -1: u_turns += 1
            current_direction = 1
        elif angle_diff < -0.5: # CCW
            ccw_dist += distances[i]
            directions_plot[i] = -1
            if current_direction == 1: u_turns += 1
            current_direction = -1

    corner_ratio = (corner_time / total_time) * 100.0 if total_time > 0 else 0.0
    dir_asymmetry = abs(cw_dist - ccw_dist) / (cw_dist + ccw_dist) if (cw_dist + ccw_dist) > 0 else 0.0

    sector_probs = sector_counts / np.sum(sector_counts)
    sector_probs = sector_probs[sector_probs > 0] 
    spatial_entropy = -np.sum(sector_probs * np.log2(sector_probs))

    # --- 7. MEANDER ---
    headings = [calculate_angle((xs[i], ys[i]), (xs[i+1], ys[i+1])) for i in range(len(xs)-1)]
    heading_diffs = [abs(normalize_angle_diff(headings[i+1] - headings[i])) for i in range(len(headings)-1)]
    meander = np.sum(heading_diffs) / total_distance if total_distance > 0 else 0.0

    # --- 8. HABITUATION ---
    minute_distances = []
    for m in range(5):
        m_start = times[0] + (m * 60.0)
        m_end = m_start + 60.0
        mask = (times[:-1] >= m_start) & (times[:-1] < m_end)
        minute_distances.append(np.sum(distances[mask]))

    habituation_index = 0.0
    if len(minute_distances) >= 5 and total_distance > 0:
        habituation_index = (minute_distances[0] - minute_distances[4]) / total_distance

    # --- 9. JSON COMPILATION ---
    features = {
        "Metadata": {
            "Analyzed_File": os.path.basename(csv_path),
            "Total_Time_Evaluated_s": round(float(total_time), 2),
            "Threshold_Type": used_threshold_type,
            "Adaptive_Threshold_Lower_px_s": round(float(t_lower), 2),
            "Adaptive_Threshold_Upper_px_s": round(float(t_upper), 2)
        },
        "I_General_Locomotor": {
            "Total_Distance_px": round(float(total_distance), 2),
            "Total_Activity_Time_s": round(safe_sum(active_phases, 'duration'), 2),
            "Total_Inactivity_Time_s": round(safe_sum(inactive_phases, 'duration'), 2),
            "Average_Speed_px_s": round(float(total_distance / total_time), 2) if total_time > 0 else 0.0,
            "Average_Activity_Speed_px_s": round(float(total_distance / safe_sum(active_phases, 'duration')), 2) if safe_sum(active_phases, 'duration') > 0 else 0.0,
            "Total_Activity_Number": len(active_phases),
            "Total_Inactivity_Number": len(inactive_phases),
            "Mean_Activity_Phase_Duration_s": round(safe_mean(active_phases, 'duration'), 2),
            "Mean_Inactivity_Phase_Duration_s": round(safe_mean(inactive_phases, 'duration'), 2)
        },
        "II_Phase_Specific": {
            "Short_Phases_LEQ_5s": {
                "Short_Inactivity_Number": len(short_inactive),
                "Short_Inactivity_Total_Time_s": round(safe_sum(short_inactive, 'duration'), 2),
                "Short_Inactivity_Mean_Time_s": round(safe_mean(short_inactive, 'duration'), 2),
                "Short_Activity_Number": len(short_active),
                "Short_Activity_Total_Time_s": round(safe_sum(short_active, 'duration'), 2),
                "Short_Activity_Mean_Time_s": round(safe_mean(short_active, 'duration'), 2),
                "Short_Activity_Total_Distance_px": round(safe_sum(short_active, 'distance'), 2),
                "Short_Activity_Mean_Distance_px": round(safe_mean(short_active, 'distance'), 2),
                "Short_Activity_Mean_Speed_px_s": round(safe_sum(short_active, 'distance') / safe_sum(short_active, 'duration'), 2) if safe_sum(short_active, 'duration') > 0 else 0.0
            },
            "Long_Phases_GT_5s": {
                "Long_Inactivity_Number": len(long_inactive),
                "Long_Inactivity_Total_Time_s": round(safe_sum(long_inactive, 'duration'), 2),
                "Long_Inactivity_Mean_Time_s": round(safe_mean(long_inactive, 'duration'), 2),
                "Long_Activity_Number": len(long_active),
                "Long_Activity_Total_Time_s": round(safe_sum(long_active, 'duration'), 2),
                "Long_Activity_Mean_Time_s": round(safe_mean(long_active, 'duration'), 2),
                "Long_Activity_Total_Distance_px": round(safe_sum(long_active, 'distance'), 2),
                "Long_Activity_Mean_Distance_px": round(safe_mean(long_active, 'distance'), 2),
                "Long_Activity_Mean_Speed_px_s": round(safe_sum(long_active, 'distance') / safe_sum(long_active, 'duration'), 2) if safe_sum(long_active, 'duration') > 0 else 0.0
            }
        },
        "III_Spatial_and_Geometric": {
            "Corner_Time_Ratio_pct": round(float(corner_ratio), 2),
            "Clockwise_Distance_px": round(float(cw_dist), 2),
            "CounterClockwise_Distance_px": round(float(ccw_dist), 2),
            "Directional_Asymmetry_Index": round(float(dir_asymmetry), 4),
            "U_Turn_Count": u_turns,
            "Spatial_Entropy_bits": round(float(spatial_entropy), 4)
        },
        "IV_Kinematic": {
            "Meander_Tortuosity_deg_px": round(float(meander), 4),
            "Acceleration_Variance_px_s2": round(float(accel_variance), 4)
        },
        "V_Pattern_Analysis_Habituation": {
            "Distance_Min_1_px": round(float(minute_distances[0]) if len(minute_distances)>0 else 0, 2),
            "Distance_Min_2_px": round(float(minute_distances[1]) if len(minute_distances)>1 else 0, 2),
            "Distance_Min_3_px": round(float(minute_distances[2]) if len(minute_distances)>2 else 0, 2),
            "Distance_Min_4_px": round(float(minute_distances[3]) if len(minute_distances)>3 else 0, 2),
            "Distance_Min_5_px": round(float(minute_distances[4]) if len(minute_distances)>4 else 0, 2),
            "Intra_Session_Habituation_Index": round(float(habituation_index), 4)
        }
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(features, f, indent=4, ensure_ascii=False)
    print(f"Success! Features extracted and saved to: {output_json_path}")

    # --- 10. GENERATE DEBUG DASHBOARD ---
    if debug_html:
        html_path = output_json_path.replace('.json', '_debug.html')
        generate_feature_dashboard(
            times, speeds, states, t_lower, t_upper, 
            xs, ys, arena_center, directions_plot, 
            minute_distances, html_path
        )


@click.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--config", default=None, type=click.Path(), help="Path to ROI config JSON file (Auto-detected if not provided)")
@click.option("--output", default=None, help="Output JSON filename (Optional)")
@click.option("--debug", is_flag=True, help="Generate a Plotly HTML dashboard to verify calculations")
def cli(csv_path, config, output, debug):
    if not os.path.exists(csv_path):
        click.echo(f"ERROR: CSV file not found: {csv_path}")
        return

    base_dir = os.path.dirname(csv_path)
    filename = os.path.splitext(os.path.basename(csv_path))[0]
    
    base_name = filename
    for suffix in ['_trajectory_filtered', '_filtered', '_trajectory']:
        if base_name.endswith(suffix): base_name = base_name.replace(suffix, '')
        
    for method in ['_simple', '_static', '_dof', '_kf', '_of', '_sub_dof_fusion', '_pure_of', '_mhi']:
        if base_name.endswith(method):
            base_name = base_name[:-len(method)]
            break

    if config is None:
        config = os.path.join(base_dir, f"{base_name}_config.json")
        if not os.path.exists(config):
            click.echo(f"ERROR: Auto-detected Config file not found: {config}")
            return
            
    if output is None:
        method_suffix = filename.replace(base_name, '').replace('_trajectory_filtered', '').replace('_trajectory', '')
        output = os.path.join(base_dir, f"{base_name}{method_suffix}_features.json")

    extract_features(csv_path, config_path=config, output_json_path=output, debug_html=debug, global_thresholds_path=None)

if __name__ == '__main__':
    cli()