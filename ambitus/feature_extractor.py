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

# ==============================================================================
# GEOMETRIC HELPERS (Ray Intersection & Line of Sight Zones)
# ==============================================================================

def geometric_median(points, eps=1e-5):
    """
    Calculates the robust geometric median (spatial median) of a set of points 
    using the Weiszfeld algorithm (iterative least squares).
    """
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

def get_arena_center(inner_poly, outer_poly):
    """
    Calculates the highly robust perspective center of the arena 
    by finding the geometric median of all inner and outer points.
    """
    # Combine all points from both polygons
    all_points = np.vstack((outer_poly, inner_poly))
    cx, cy = geometric_median(all_points)
    return (int(cx), int(cy))

def get_poly_centroid(poly):
    """ Calculates the exact center of mass for a polygon (for labeling). """
    M = cv2.moments(poly)
    if M["m00"] != 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return (int(poly[0][0]), int(poly[0][1]))

def get_ray_intersection(A, B, outer_poly):
    """
    Shoots a mathematical ray from A through B, and finds where it precisely 
    intersects the outer_poly. This extends the inner walls to the outer walls.
    """
    x1, y1 = A
    x2, y2 = B
    
    best_t = float('inf')
    best_P = None
    
    n = len(outer_poly)
    for i in range(n):
        x3, y3 = outer_poly[i]
        x4, y4 = outer_poly[(i+1)%n]
        
        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x4 - x3
        dy2 = y4 - y3
        
        det = dx1 * (-dy2) - dy1 * (-dx2)
        if abs(det) < 1e-6: continue # Parallel lines
        
        cx_ax = x3 - x1
        cy_ay = y3 - y1
        
        t = (cx_ax * (-dy2) - cy_ay * (-dx2)) / det
        u = (dx1 * cy_ay - dy1 * cx_ax) / det
        
        # t must be > 0 (forward direction of ray)
        # u must be between 0 and 1 (must hit the actual segment of outer_poly)
        if t > 0.001 and -0.001 <= u <= 1.001:
            if t < best_t:
                best_t = t
                best_P = (x1 + t * dx1, y1 + t * dy1)
                
    if best_P is not None:
        return best_P
    return B # Fallback safety

def clean_poly(poly):
    poly = np.array(poly, dtype=float)
    if len(poly) > 1 and np.linalg.norm(poly[0] - poly[-1]) < 1.0:
        poly = poly[:-1]
    return poly

def build_arena_zones(inner_poly, outer_poly):
    """
    Creates 8 exact geometric zones (4 corners, 4 corridors) based on the 
    Line-of-Sight rules requested.
    """
   
    zones = []
    n = len(inner_poly)
    
    # Calculate all forward and backward intersections for the inner corners
    R_F = [] # Forward rays
    R_B = [] # Backward rays
    for i in range(n):
        prev_i = (i - 1) % n
        next_i = (i + 1) % n
        
        # Ray extending the incoming wall forward
        r_f = get_ray_intersection(inner_poly[prev_i], inner_poly[i], outer_poly)
        R_F.append(r_f)
        
        # Ray extending the outgoing wall backward
        r_b = get_ray_intersection(inner_poly[next_i], inner_poly[i], outer_poly)
        R_B.append(r_b)
        
    for i in range(n):
        next_i = (i + 1) % n
        
        # 1. CORRIDOR (Between inner point i and next_i)
        corridor_poly = np.array([
            inner_poly[i],
            inner_poly[next_i],
            R_B[next_i],
            R_F[i]
        ], dtype=np.int32)
        
        zones.append({
            "id": i * 2,
            "poly": corridor_poly,
            "is_corner": False,
            "label": f"Corridor {i}"
        })
        
        # 2. CORNER (At inner point next_i)
        corner_poly = np.array([
            inner_poly[next_i],
            R_F[next_i],
            outer_poly[next_i], 
            R_B[next_i]
        ], dtype=np.int32)
        
        zones.append({
            "id": i * 2 + 1,
            "poly": corner_poly,
            "is_corner": True,
            "label": f"Corner {next_i}"
        })
        
    return zones

# For kinematic (CW/CCW) features
def calculate_angle(p1, p2):
    return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

def normalize_angle_diff(diff):
    while diff <= -180: diff += 360
    while diff > 180: diff -= 360
    return diff

# ==============================================================================
# DEBUG VISUALIZATION
# ==============================================================================
def generate_zone_debug_image(config_data, video_path, output_image_path):
    """
    Draws the mathematically calculated corner and corridor zones 
    perfectly overlaid on the video frame.
    """
    inner_poly = np.array(config_data["inner_poly"], dtype=np.int32)
    outer_poly = np.array(config_data["outer_poly"], dtype=np.int32)
    inner_clean = clean_poly(inner_poly)
    outer_clean = clean_poly(outer_poly)

    # --- load frame ---
    img = None
    if video_path and os.path.exists(video_path):
        is_mpg = video_path.lower().endswith('.mpg')
        if is_mpg:
            reader = imageio.get_reader(video_path, 'ffmpeg', input_params=[], output_params=['-vf', 'yadif,scale=768:576,setsar=1'])
            frame = reader.get_data(0)
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            reader.close()
        else:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret: img = frame
            cap.release()

    if img is None:
        max_x = np.max(outer_poly[:, 0]) + 100
        max_y = np.max(outer_poly[:, 1]) + 100
        img = np.zeros((max_y, max_x, 3), dtype=np.uint8)

    # --- generate the precise zones ---
    zones = build_arena_zones(inner_clean, outer_clean)

    overlay = img.copy()

    # Draw colored fills
    for z in zones:
        color = (0, 0, 255) if z["is_corner"] else (0, 255, 0)
        cv2.fillPoly(overlay, [z["poly"]], color)

    # Blend
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Draw exact boundaries and labels
    for z in zones:
        color = (0, 0, 255) if z["is_corner"] else (0, 255, 0)
        cv2.polylines(img, [z["poly"]], True, color, 2)

    cv2.polylines(img, [outer_poly], True, (255, 255, 255), 2)
    cv2.polylines(img, [inner_poly], True, (200, 200, 200), 2)

    cv2.imwrite(output_image_path, img)
    print(f"Geometric Zone debug image saved to: {output_image_path}")

def generate_feature_dashboard(times, speeds, states, xs, ys, arena_center, directions, minute_distances, html_path):
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

    fig.add_trace(go.Scatter(x=times[:-1], y=speeds, mode='lines', line=dict(color='black', width=1), name='Speed (px/s)'), row=1, col=1)

    legend_flags = {2: False, 1: False, 0: False}
    start_idx = 0
    for key, group in groupby(states):
        length = sum(1 for _ in group)
        end_idx = min(start_idx + length, len(times) - 2)
        
        t0, t1 = times[start_idx], times[end_idx]
        
        if key == 2:
            color, name = "rgba(0, 255, 0, 0.2)", "High Active (Locomotion)"
        elif key == 1:
            color, name = "rgba(255, 255, 0, 0.3)", "Low Active (Micro-movement)"
        else:
            color, name = "rgba(255, 0, 0, 0.2)", "Inactive (Resting)"
            
        show_leg = not legend_flags[key]
        legend_flags[key] = True
            
        fig.add_vrect(x0=t0, x1=t1, fillcolor=color, layer="below", line_width=0, row=1, col=1)
        
        if show_leg:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=10, color=color.replace('0.2', '0.8').replace('0.3', '0.8'), symbol='square'),
                name=name
            ), row=1, col=1)
            
        start_idx += length

    cw_x, cw_y, ccw_x, ccw_y = [], [], [], []
    for i in range(len(directions)-1): # Safe loop
        if directions[i] == 1:
            cw_x.append(xs[i]); cw_y.append(ys[i])
        elif directions[i] == -1:
            ccw_x.append(xs[i]); ccw_y.append(ys[i])
            
    fig.add_trace(go.Scatter(x=cw_x, y=cw_y, mode='markers', marker=dict(color='blue', size=3), name='CW Move'), row=2, col=1)
    fig.add_trace(go.Scatter(x=ccw_x, y=ccw_y, mode='markers', marker=dict(color='orange', size=3), name='CCW Move'), row=2, col=1)
    fig.add_trace(go.Scatter(x=[arena_center[0]], y=[arena_center[1]], mode='markers', marker=dict(color='black', symbol='cross', size=10), name='Arena Center'), row=2, col=1)

    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, row=2, col=1)

    mins = [f"Min {i+1}" for i in range(len(minute_distances))]
    fig.add_trace(go.Bar(x=mins, y=minute_distances, marker_color='purple', name='Distance/Min'), row=2, col=2)

    max_time = times[-1] if len(times) > 0 else 300
    fig.update_xaxes(range=[0, max_time + 5], row=1, col=1)
    
    fig.update_layout(
        height=800, width=1200, 
        title_text="Feature Extraction Debug Dashboard", 
        showlegend=True, hovermode="closest"
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
    inner_poly = clean_poly(inner_poly)
    outer_poly = clean_poly(outer_poly)
    arena_center = get_arena_center(inner_poly, outer_poly)

    # --- 2. BUILD PHYSICAL ZONES ---
    zones = build_arena_zones(inner_poly, outer_poly)

    # --- 3. LOAD DATA (Now expecting State column!) ---
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

    # --- 4. KINEMATICS ---
    dt = np.diff(times)  
    dx = np.diff(xs)
    dy = np.diff(ys)
    
    safe_dt = np.where(dt == 0, 0.001, dt)
    distances = np.sqrt(dx**2 + dy**2)
    speeds = distances / safe_dt
    
    accelerations = np.diff(speeds) / safe_dt[:-1]
    accel_variance = np.var(accelerations) if len(accelerations) > 0 else 0.0

    total_distance = np.sum(distances)

    # --- 5. PHASE STABILIZATION (1s Majority Voting for STATES) ---
    avg_dt = total_time / len(times) if len(times) > 0 else 0.04
    fps = int(1.0 / avg_dt) if avg_dt > 0 else 25
    
    one_second_states = np.ones(len(speeds)//fps + 1, dtype=int) * -1

    for i in range(len(one_second_states)):
        start = i * fps
        end = min((i + 1) * fps, len(speeds))
        if start >= end: break
        
        window_states = raw_states[start:end]
        if len(window_states) == 0:
            one_second_states[i] = 0
            continue
        
        high_active_count = np.sum(window_states == 2)
        low_active_count = np.sum(window_states == 1)
        inactive_count = np.sum(window_states == 0)
        
        if high_active_count > low_active_count and high_active_count > inactive_count:
            one_second_states[i] = 2  
        elif low_active_count > high_active_count and low_active_count > inactive_count:
            one_second_states[i] = 1  
        elif high_active_count + low_active_count > inactive_count:
            one_second_states[i] = 1  
        else:
            one_second_states[i] = 0  

    smoothed_states = np.zeros(len(speeds), dtype=int)
    for i in range(len(one_second_states)):
        start = i * fps
        end = min((i + 1) * fps, len(speeds))
        if start < end:
            smoothed_states[start:end] = one_second_states[i]

    states = smoothed_states

    # --- 6. PHASE ANALYSIS ---
    high_active_phases, low_active_phases, inactive_phases = [], [], []
    
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

    # --- 7. DIRECTIONAL STABILIZATION (1s Majority Voting for CW/CCW) ---
    raw_directions = np.zeros(len(xs), dtype=int)
    angles_to_center = [calculate_angle(arena_center, (x, y)) for x, y in zip(xs, ys)]
    
    # Kiszámoljuk nyersen pontról-pontra
    for i in range(len(xs) - 1):
        angle_diff = normalize_angle_diff(angles_to_center[i+1] - angles_to_center[i])
        if angle_diff > 0.5:
            raw_directions[i] = 1   # CW
        elif angle_diff < -0.5:
            raw_directions[i] = -1  # CCW
            
    # Copy for the last element
    if len(raw_directions) > 1:
        raw_directions[-1] = raw_directions[-2]

    # Csináljuk meg a többségi szavazást 1 másodperces (fps) ablakokra
    smoothed_directions = np.zeros(len(xs), dtype=int)
    for i in range((len(xs) // fps) + 1):
        start = i * fps
        end = min((i + 1) * fps, len(xs))
        if start >= end: break
        
        window_dirs = raw_directions[start:end]
        cw_count = np.sum(window_dirs == 1)
        ccw_count = np.sum(window_dirs == -1)
        
        if cw_count > ccw_count:
            dom_dir = 1
        elif ccw_count > cw_count:
            dom_dir = -1
        else:
            dom_dir = 0
            
        smoothed_directions[start:end] = dom_dir

    # ÚJ SZABÁLY: Csak akkor érvényes az irány, ha a patkány High Active (State == 2)
    for i in range(len(smoothed_directions)):
        # Biztonságos indexelés: ha az 'i' nagyobb, mint a states hossza, az utolsó elemet nézi
        state_idx = min(i, len(states) - 1)
        if states[state_idx] != 2:
            smoothed_directions[i] = 0

    # MOST számítjuk az U-Turn-öket és távolságokat a KISIMÍTOTT és SZŰRT adatokból!
    cw_dist, ccw_dist = 0.0, 0.0
    u_turns = 0
    last_valid_dir = 0
    
    for i in range(len(xs) - 1):
        d = smoothed_directions[i]
        if d == 1:
            cw_dist += distances[i]
            if last_valid_dir == -1: 
                u_turns += 1
            last_valid_dir = 1
        elif d == -1:
            ccw_dist += distances[i]
            if last_valid_dir == 1: 
                u_turns += 1
            last_valid_dir = -1

    dir_asymmetry = abs(cw_dist - ccw_dist) / (cw_dist + ccw_dist) if (cw_dist + ccw_dist) > 0 else 0.0

    # --- 8. SPATIAL ANALYSIS (Corners vs Corridors) ---
    corner_time = 0.0
    sector_counts = np.zeros(8, dtype=int)
    corner_flags = np.zeros(len(xs), dtype=int)

    for i in range(len(xs) - 1):
        pt = (float(xs[i]), float(ys[i]))
        is_corner = False
        sec_id = -1
        
        for z in zones:
            if cv2.pointPolygonTest(z["poly"], pt, False) >= 0:
                sec_id = z["id"]
                is_corner = z["is_corner"]
                break
                
        if sec_id != -1:
            sector_counts[sec_id] += 1
            if is_corner: corner_time += dt[i]

        corner_flags[i] = 1 if is_corner else 0

    corner_ratio = (corner_time / total_time) * 100.0 if total_time > 0 else 0.0
    
    sector_probs = sector_counts / np.sum(sector_counts)
    sector_probs = sector_probs[sector_probs > 0] 
    spatial_entropy = -np.sum(sector_probs * np.log2(sector_probs))

    # --- 9. MEANDER ---
    headings = [calculate_angle((xs[i], ys[i]), (xs[i+1], ys[i+1])) for i in range(len(xs)-1)]
    heading_diffs = [abs(normalize_angle_diff(headings[i+1] - headings[i])) for i in range(len(headings)-1)]
    meander = np.sum(heading_diffs) / total_distance if total_distance > 0 else 0.0

    # --- 10. HABITUATION ---
    minute_distances = []
    for m in range(5):
        m_start = times[0] + (m * 60.0)
        m_end = m_start + 60.0
        mask = (times[:-1] >= m_start) & (times[:-1] < m_end)
        minute_distances.append(np.sum(distances[mask]))

    habituation_index = 0.0
    if len(minute_distances) >= 5 and total_distance > 0:
        habituation_index = (minute_distances[0] - minute_distances[4]) / total_distance

    # Mentés a videós lejátszáshoz (Itt már a SIMÍTOTT irányt és sarkot mentjük!)
    debug_csv_path = output_json_path.replace('.json', '_debug.csv')
    with open(debug_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time', 'X', 'Y', 'Speed', 'State', 'Direction', 'Is_Corner'])
        for i in range(len(times)-1):
            writer.writerow([times[i], xs[i], ys[i], speeds[i], states[i], smoothed_directions[i], corner_flags[i]])

    # --- 11. JSON COMPILATION ---
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

    # --- 12. GENERATE DEBUG DASHBOARD ---
    if debug_html:
        print("Generating debug dashboards...")
        html_path = output_json_path.replace('.json', '_debug.html')
        generate_feature_dashboard(
            times, speeds, states, 
            xs, ys, arena_center, smoothed_directions, 
            minute_distances, html_path
        )
        
        img_path = output_json_path.replace('.json', '_sectors_debug.jpg')
        if video_path is not None:
            generate_zone_debug_image(config_data, video_path, img_path)


@click.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--config", default=None, type=click.Path(), help="Path to ROI config JSON file")
@click.option("--video", default=None, type=click.Path(), help="Path to original video file for generating the sector debug image")
@click.option("--output", default=None, help="Output JSON filename")
@click.option("--debug", is_flag=True, help="Generate a Plotly HTML dashboard and Sector image")
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