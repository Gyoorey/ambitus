import os
import csv
import json
import numpy as np
import click
import cv2
import imageio
from itertools import groupby

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_sector_debug_image(config_data, video_path, output_image_path):
    """
    Generates an image showing the first frame of the video (if available) 
    overlaid with the arena polygons and the calculated sector boundaries.
    """
    inner_poly = np.array(config_data["inner_poly"], dtype=np.int32)
    outer_poly = np.array(config_data["outer_poly"], dtype=np.int32)
    # arena_center = get_polygon_centroid(inner_poly)
    arena_center = get_polygon_center(inner_poly)
    
    # Calculate phi exactly as in the extraction engine
    upper_right_corner = outer_poly[3]
    phi = calculate_angle(arena_center, upper_right_corner) - 45

    img = None
    if video_path and os.path.exists(video_path):

        is_mpg = video_path.lower().endswith('.mpg')
    
        if is_mpg:
            input_params = []
            output_params = ['-vf', 'yadif,scale=768:576,setsar=1']
            reader = imageio.get_reader(video_path, 'ffmpeg', input_params=input_params, output_params=output_params)
            # Read the first frame (index 0)
            frame = reader.get_data(0)
            # imageio returns RGB, but cv2 expects BGR for drawing and saving
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            reader.close()
        else:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                img = frame
            cap.release()
        
    if img is None:
        # Create a blank black canvas if video is not available
        max_x = np.max(outer_poly[:, 0]) + 100
        max_y = np.max(outer_poly[:, 1]) + 100
        img = np.zeros((max_y, max_x, 3), dtype=np.uint8)
        print("WARNING: Could not read video frame. Drawing sectors on a blank canvas.")

    # Calculate Dynamic Sectors based on outer_poly corners
    perspective_center, dynamic_sectors = setup_dynamic_sectors(inner_poly, outer_poly)
    cx, cy = perspective_center
    # cx, cy = arena_center

    # Draw Polygons
    cv2.polylines(img, [outer_poly], isClosed=True, color=(255, 255, 255), thickness=2)
    cv2.polylines(img, [inner_poly], isClosed=True, color=(200, 200, 200), thickness=1, lineType=cv2.LINE_AA)
    
    # Draw Center
    cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

    # Calculate max radius for drawing lines
    max_r = max([np.hypot(p[0]-cx, p[1]-cy) for p in outer_poly]) * 1.1

    # Draw the dynamic sector boundaries
    for sec in dynamic_sectors:
        # Draw boundary line (starting edge of the sector)
        theta_rad = np.radians(sec['start'])
        end_x = int(cx + max_r * np.cos(theta_rad))
        end_y = int(cy + max_r * np.sin(theta_rad))
        cv2.line(img, (cx, cy), (end_x, end_y), (0, 0, 255), 2)
        
        # Label the sector in the middle of its span
        mid_angle = sec['start'] + (sec['end'] - sec['start']) / 2
        if sec['end'] < sec['start']: # Handle 360 wrap
            mid_angle = sec['start'] + (sec['end'] + 360 - sec['start']) / 2
        mid_angle %= 360
        
        label_theta_rad = np.radians(mid_angle)
        text_x = int(cx + (max_r * 0.6) * np.cos(label_theta_rad))
        text_y = int(cy + (max_r * 0.6) * np.sin(label_theta_rad))
        
        label = f"S{sec['id']}: {'Corner' if sec['is_corner'] else 'Corridor'}"
        color = (0, 165, 255) if sec['is_corner'] else (255, 255, 0)
        cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.imwrite(output_image_path, img)
    print(f"Sector debug image saved to: {output_image_path}")

# ==============================================================================
# GEOMETRIC HELPERS
# ==============================================================================
def get_polygon_centroid(poly):
    M = cv2.moments(poly)
    if M["m00"] != 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return (0, 0)

def get_polygon_center(poly):
    """ 
    Calculates the center of a polygon using its bounding box.
    """
    x, y, w, h = cv2.boundingRect(poly)
    return (int(x + w / 2.0), int(y + h / 2.0))

def geometric_median(points, eps=1e-5):
    points = np.array(points, dtype=float)
    guess = np.mean(points, axis=0)

    while True:
        distances = np.linalg.norm(points - guess, axis=1)

        nonzero = distances > 1e-10
        if not np.any(nonzero):
            return guess

        weights = 1 / distances[nonzero]
        new_guess = np.sum(points[nonzero] * weights[:, None], axis=0) / np.sum(weights)

        if np.linalg.norm(new_guess - guess) < eps:
            return new_guess

        guess = new_guess

def setup_dynamic_sectors(inner_poly, outer_poly):
    """
    Calculates the perspective center using diagonal intersections and 
    creates dynamically sized sectors based on the real corner angles.
    """

    def clean(poly):
        poly = np.array(poly, dtype=float)
        if len(poly) > 1 and np.array_equal(poly[0], poly[-1]):
            poly = poly[:-1]
        return poly

    outer = clean(outer_poly)
    inner = clean(inner_poly)
    all_points = np.vstack((outer, inner))
    cx, cy = geometric_median(all_points)
       
    sorted_inner = sorted(
        inner,
        key=lambda p: np.arctan2(p[1] - cy, p[0] - cx)
    )

    # --- Compute corner angles ---
    corner_angles = []
    for p in sorted_inner:
        angle = np.degrees(np.arctan2(p[1] - cy, p[0] - cx))
        angle = (angle + 360) % 360
        corner_angles.append(angle)
    
    boundaries = []
    corner_ratio = 0.2  # 20% corner
    for i in range(4):
        a_curr = corner_angles[i]
        a_next = corner_angles[(i + 1) % 4]

        delta = (a_next - a_curr) % 360

        corridor_end = (a_next - corner_ratio * delta) % 360
        corridor_start = (a_curr + corner_ratio * delta) % 360

        boundaries.extend([corridor_start, corridor_end])

        
    # 5. Define sector properties
    sectors = [
        {'start': boundaries[0], 'end': boundaries[1], 'id': 0, 'is_corner': False}, # Corridor 0
        {'start': boundaries[1], 'end': boundaries[2], 'id': 1, 'is_corner': True},  # Corner 1
        {'start': boundaries[2], 'end': boundaries[3], 'id': 2, 'is_corner': False}, # Corridor 1
        {'start': boundaries[3], 'end': boundaries[4], 'id': 3, 'is_corner': True},  # Corner 2
        {'start': boundaries[4], 'end': boundaries[5], 'id': 4, 'is_corner': False}, # Corridor 2
        {'start': boundaries[5], 'end': boundaries[6], 'id': 5, 'is_corner': True},  # Corner 3
        {'start': boundaries[6], 'end': boundaries[7], 'id': 6, 'is_corner': False}, # Corridor 3
        {'start': boundaries[7], 'end': boundaries[0], 'id': 7, 'is_corner': True},  # Corner 0
    ]
    
    # integer center for drawing purposes
    return (int(cx), int(cy)), sectors

def get_sector_dynamic(x, y, cx, cy, sectors):
    angle = (np.degrees(np.arctan2(y - cy, x - cx)) + 360) % 360
    for sec in sectors:
        s, e = sec['start'], sec['end']
        if s < e:
            if s <= angle < e: return sec['id'], sec['is_corner']
        else: # Wraps around 360 (usually Sector 7)
            if angle >= s or angle < e: return sec['id'], sec['is_corner']
    return 0, False

def calculate_angle(p1, p2):
    return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

def normalize_angle_diff(diff):
    while diff <= -180: diff += 360
    while diff > 180: diff -= 360
    return diff

def get_sector(x, y, cx, cy, phi=0):
    angle = np.degrees(np.arctan2(y - cy, x - cx)) + phi
    if angle < 0: angle += 360  
    shifted = (angle + 22.5) % 360
    sector_id = int(shifted // 45)
    is_corner = (sector_id % 2 != 0)
    return sector_id, is_corner

# ==============================================================================
# DEBUG VISUALIZATION (PLOTLY)
# ==============================================================================
def generate_feature_dashboard(times, speeds, states, 
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
        shared_xaxes=False,  
        shared_yaxes=False   
    )

    # --- PLOT 1: SPEED & STATES ---
    fig.add_trace(go.Scatter(
        x=times[:-1], y=speeds, mode='lines', line=dict(color='black', width=1), name='Speed (px/s)'
    ), row=1, col=1)

    legend_flags = {2: False, 1: False, 0: False}

    start_idx = 0
    for key, group in groupby(states):
        length = sum(1 for _ in group)
        end_idx = min(start_idx + length, len(times) - 2)
        
        t0 = times[start_idx]
        t1 = times[end_idx]
        
        if key == 2:
            color = "rgba(0, 255, 0, 0.2)" # Green
            name = "High Active (Locomotion)"
        elif key == 1:
            color = "rgba(255, 255, 0, 0.3)" # Yellow
            name = "Low Active (Micro-movement)"
        else:
            color = "rgba(255, 0, 0, 0.2)" # Red
            name = "Inactive (Resting)"
            
        show_leg = not legend_flags[key]
        legend_flags[key] = True
            
        fig.add_vrect(
            x0=t0, x1=t1, fillcolor=color, layer="below", line_width=0, 
            row=1, col=1
        )
        
        # Dummy trace for legend
        if show_leg:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=10, color=color.replace('0.2', '0.8').replace('0.3', '0.8'), symbol='square'),
                name=name
            ), row=1, col=1)
            
        start_idx += length

    # --- PLOT 2: SPATIAL MAP & DIRECTIONS ---
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

    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, row=2, col=1)

    # --- PLOT 3: HABITUATION BAR CHART ---
    mins = [f"Min {i+1}" for i in range(len(minute_distances))]
    fig.add_trace(go.Bar(x=mins, y=minute_distances, marker_color='purple', name='Distance/Min'), row=2, col=2)

    # --- LAYOUT & AXES SETTINGS ---
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
        title_text="Feature Extraction Debug Dashboard (1s Smoothed States)", 
        showlegend=True, 
        hovermode="closest"
    )
    
    fig.write_html(html_path)
    print(f"Debug Dashboard saved to: {html_path}")

# ==============================================================================
# FEATURE EXTRACTION ENGINE
# ==============================================================================
def extract_features(csv_path, config_path, output_json_path, debug_html=False, video_path=None):
    if not os.path.exists(csv_path) or not os.path.exists(config_path):
        print(f"ERROR: Missing input files for extraction.")
        return

    # --- 1. LOAD CONFIG ---
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    inner_poly = np.array(config_data["inner_poly"], dtype=np.int32)
    outer_poly = np.array(config_data["outer_poly"], dtype=np.int32)
    # arena_center = get_polygon_centroid(inner_poly)
    arena_center = get_polygon_center(inner_poly)

    # --- 2. LOAD DATA (Now expecting State column!) ---
    times, xs, ys, raw_states = [], [], [], []
    has_state_col = False
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader, None)
        
        if header and len(header) >= 4 and header[3].lower() == 'state':
            has_state_col = True
            
        for row in reader:
            try:
                times.append(float(row[0]))
                xs.append(float(row[1]))
                ys.append(float(row[2]))
                
                if has_state_col:
                    raw_states.append(int(row[3]))
                else:
                    raw_states.append(0) # Fallback
            except (ValueError, IndexError):
                pass
                
    if len(times) < 10: 
        print("ERROR: Not enough data points.")
        return

    times, xs, ys, raw_states = np.array(times), np.array(xs), np.array(ys), np.array(raw_states)
    total_time = times[-1] - times[0]

    if not has_state_col:
        print("WARNING: 'State' column not found in CSV! Classification features will be 0.")

    # --- 3. KINEMATICS ---
    dt = np.diff(times)  
    dx = np.diff(xs)
    dy = np.diff(ys)
    
    # Safe division
    safe_dt = np.where(dt == 0, 0.001, dt)
    distances = np.sqrt(dx**2 + dy**2)
    speeds = distances / safe_dt
    
    accelerations = np.diff(speeds) / safe_dt[:-1]
    accel_variance = np.var(accelerations) if len(accelerations) > 0 else 0.0

    total_distance = np.sum(distances)

    # --- 4. PHASE STABILIZATION (1s Majority Voting) ---
    avg_dt = total_time / len(times) if len(times) > 0 else 0.04
    fps = int(1.0 / avg_dt) if avg_dt > 0 else 25
    
    one_second_states = np.ones(len(speeds)//fps, dtype=int) * -1

    for i in range(len(one_second_states)):
        start = i * fps
        end = min((i + 1) * fps, len(speeds))
        window_states = raw_states[start:end]
        
        if len(window_states) == 0:
            one_second_states[i] = 0
            continue
        
        high_active_count = np.sum(window_states == 2)
        low_active_count = np.sum(window_states == 1)
        inactive_count = np.sum(window_states == 0)
        
        # Tie-breaking towards low activity, then inactivity
        if high_active_count > low_active_count and high_active_count > inactive_count:
            one_second_states[i] = 2  # High Active
        elif low_active_count > high_active_count and low_active_count > inactive_count:
            one_second_states[i] = 1  # Low Active
        elif high_active_count + low_active_count > inactive_count:
            one_second_states[i] = 1  # Mixed but overall active -> Low Active
        else:
            one_second_states[i] = 0  # Inactive

    # Convert back to frame-level states
    smoothed_states = np.zeros(len(speeds), dtype=int)
    for i in range(len(one_second_states)):
        start = i * fps
        end = min((i + 1) * fps, len(speeds))
        smoothed_states[start:end] = one_second_states[i]
        
    if end < len(speeds):
        smoothed_states[end:] = one_second_states[-1] if len(one_second_states) > 0 else 0

    states = smoothed_states

    # --- 5. PHASE ANALYSIS ---
    high_active_phases = []
    low_active_phases = []
    inactive_phases = []
    
    start_idx = 0
    for key, group in groupby(states):
        length = sum(1 for _ in group)
        end_idx = min(start_idx + length, len(times) - 1)
        phase_duration = times[end_idx] - times[start_idx]
        phase_dist = np.sum(distances[start_idx:end_idx])
        
        if key == 2:
            high_active_phases.append({'duration': phase_duration, 'distance': phase_dist})
        elif key == 1:
            low_active_phases.append({'duration': phase_duration, 'distance': phase_dist})
        else:
            inactive_phases.append({'duration': phase_duration})
            
        start_idx += length

    # Combined activity
    all_active_phases = high_active_phases + low_active_phases

    short_active = [p for p in all_active_phases if p['duration'] <= 5.0]
    long_active = [p for p in all_active_phases if p['duration'] > 5.0]
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

    total_high_active_time = safe_sum(high_active_phases, 'duration')
    total_low_active_time = safe_sum(low_active_phases, 'duration')
    total_inactive_time = safe_sum(inactive_phases, 'duration')

    # --- 6. SPATIAL & DIRECTIONAL ANALYSIS ---
    corner_time = 0.0
    cw_dist, ccw_dist = 0.0, 0.0
    u_turns = 0
    
    # Initialize the new dynamic sectors
    perspective_center, dynamic_sectors = setup_dynamic_sectors(inner_poly, outer_poly)
    cx, cy = perspective_center

    # Compute raw angles from the perspective center (no phi offset needed anymore)
    angles_to_center = [(np.degrees(np.arctan2(y - cy, x - cx)) + 360) % 360 for x, y in zip(xs, ys)]
    
    sector_counts = np.zeros(8, dtype=int)
    current_direction = 0 
    directions_plot = np.zeros(len(xs)-1, dtype=int)
    
    for i in range(len(xs) - 1):
        # Use the dynamic sector logic
        sec_id, is_corner = get_sector_dynamic(xs[i], ys[i], cx, cy, dynamic_sectors)
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

    # Save the smoothed states and the angles for the debug csv
    debug_csv_path = output_json_path.replace('.json', '_debug.csv')
    with open(debug_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time', 'X', 'Y', 'Speed', 'State', 'Angle_to_Center'])
        for i in range(len(times)-1):
            writer.writerow([times[i], xs[i], ys[i], speeds[i], states[i], angles_to_center[i]])
    print(f"Debug CSV with smoothed 1s states and directions saved to: {debug_csv_path}")

    # --- 9. JSON COMPILATION ---
    features = {
        "Metadata": {
            "Analyzed_File": os.path.basename(csv_path),
            "Total_Time_Evaluated_s": round(float(total_time), 2),
            "State_Classification_Source": "Pre-classified CSV (Video Pixel Variance)" if has_state_col else "ERROR: Missing"
        },
        "I_General_Locomotor": {
            "Total_Distance_px": round(float(total_distance), 2),
            "Total_High_Activity_Time_s": round(total_high_active_time, 2),
            "Total_Low_Activity_Time_s": round(total_low_active_time, 2),
            "Total_Combined_Activity_Time_s": round(total_high_active_time + total_low_active_time, 2),
            "Total_Inactivity_Time_s": round(total_inactive_time, 2),
            "Average_Speed_px_s": round(float(total_distance / total_time), 2) if total_time > 0 else 0.0,
            "Average_Activity_Speed_px_s": round(float(total_distance / safe_sum(all_active_phases, 'duration')), 2) if safe_sum(all_active_phases, 'duration') > 0 else 0.0,
            "Total_Activity_Number": len(all_active_phases),
            "Total_Inactivity_Number": len(inactive_phases),
            "Mean_Activity_Phase_Duration_s": round(safe_mean(all_active_phases, 'duration'), 2),
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
            "Corner_Time_s": round(float(corner_time), 2),
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
        print("Generating debug dashboard...")
        html_path = output_json_path.replace('.json', '_debug.html')
        generate_feature_dashboard(
            times, speeds, states, 
            xs, ys, arena_center, directions_plot, 
            minute_distances, html_path
        )
        # Generate the sector alignment image
        img_path = output_json_path.replace('.json', '_sectors_debug.jpg')
        if video_path is not None:
            generate_sector_debug_image(config_data, video_path, img_path)


@click.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--config", default=None, type=click.Path(), help="Path to ROI config JSON file (Auto-detected if not provided)")
@click.option("--video", default=None, type=click.Path(), help="Path to original video file for generating the sector debug image")
@click.option("--output", default=None, help="Output JSON filename (Optional)")
@click.option("--debug", is_flag=True, help="Generate a Plotly HTML dashboard to verify calculations")
def cli(csv_path, config, video, output, debug):
    if not os.path.exists(csv_path):
        click.echo(f"ERROR: CSV file not found: {csv_path}")
        return

    base_dir = os.path.dirname(csv_path)
    filename = os.path.splitext(os.path.basename(csv_path))[0]
    
    base_name = filename
    for suffix in ['_trajectory_filtered_classified', '_classified', '_trajectory_filtered', '_filtered', '_trajectory']:
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
        method_suffix = filename.replace(base_name, '').replace('_trajectory_filtered_classified', '').replace('_trajectory_filtered', '').replace('_trajectory', '')
        output = os.path.join(base_dir, f"{base_name}{method_suffix}_features.json")

    extract_features(csv_path, config_path=config, output_json_path=output, debug_html=debug, video_path=video)

if __name__ == '__main__':
    cli()