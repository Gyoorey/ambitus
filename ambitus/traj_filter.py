import csv
import os
import click
import numpy as np
import cv2
import json
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from itertools import groupby
from operator import itemgetter

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# GEOMETRIC VALIDATION
# ==============================================================================
def is_path_valid(p1, p2, inner_poly):
    """
    Checks if a straight line between p1 and p2 crosses the forbidden inner zone.
    Samples 10 points along the line. If any point is inside the inner polygon,
    the path is physically impossible (rat jumped through a wall).
    """
    if inner_poly is None or len(inner_poly) == 0:
        return True  # If no polygon is provided, all paths are considered valid
        
    x1, y1 = p1
    x2, y2 = p2
    
    # Sample points along the line
    for i in range(1, 10):
        t = i / 10.0
        test_x = x1 + t * (x2 - x1)
        test_y = y1 + t * (y2 - y1)
        
        # cv2.pointPolygonTest returns > 0 if the point is strictly INSIDE the polygon
        if cv2.pointPolygonTest(inner_poly, (float(test_x), float(test_y)), False) > 0:
            return False
            
    return True

# ==============================================================================
# INSERTION DETECTION (Start / End Cropping)
# ==============================================================================
def detect_insertion_time(times, xs, ys, window_sec=5.0, max_gap_sec=1.0, min_movement=15.0):
    """
    Finds the exact moment the rat is placed in the arena by looking for the first
    continuous tracking period.
    - window_sec: Look for a stable window of this many seconds.
    - max_gap_sec: The tracker must not lose the rat for more than this time within the window.
    - min_movement: The object must move at least this many pixels (ignores static false positives).
    """
    for i in range(len(times)):
        t_start = times[i]
        t_end_target = t_start + window_sec

        # Find all points within this time window
        mask = (times >= t_start) & (times <= t_end_target)
        t_window = times[mask]

        # Check if the window actually covers the required time span
        if len(t_window) > 1 and (t_window[-1] - t_window[0]) >= (window_sec - 0.5):
            
            # Check for large gaps (e.g. tracker lost object because researcher hand left)
            max_gap = np.max(np.diff(t_window))
            if max_gap <= max_gap_sec:
                
                # Check if it actually moves (not a static cable/shadow)
                x_win = xs[mask]
                y_win = ys[mask]
                movement = max(np.max(x_win) - np.min(x_win), np.max(y_win) - np.min(y_win))

                if movement >= min_movement:
                    return t_start
                    
    return times[0] # Fallback if no clean window is found

# ==============================================================================
# VISUALIZATION (PLOTLY & CV2)
# ==============================================================================
def draw_plotly_plot(times, xs, ys, uniform_times, x_filtered, y_filtered, filename=None, show_plot=False):
    """
    Creates an interactive dashboard using Plotly.
    Top Row: X and Y Coordinates over Time.
    Bottom Row: 2D Trajectory Map (Top-Down).
    """
    # 2 rows, 2 columns grid. The 1st column in the 2nd row spans across both columns (colspan=2)
    fig = make_subplots(
        rows=2, cols=2, 
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=("X Coordinate over Time", "Y Coordinate over Time", "2D Trajectory Map (Top-Down)"),
        row_heights=[0.4, 0.6] # Top graphs get 40%, bottom map gets 60% height
    )

    # --- PLOT 1: X OVER TIME (Row 1, Col 1) ---
    fig.add_trace(go.Scatter(
        x=times, y=xs, mode='markers', 
        marker=dict(color='rgba(255, 100, 100, 0.5)', size=5), 
        name='Raw Data (X)'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=uniform_times, y=x_filtered, mode='lines', 
        line=dict(color='blue', width=2), 
        connectgaps=False,
        name='Filtered (X)'
    ), row=1, col=1)

    # --- PLOT 2: Y OVER TIME (Row 1, Col 2) ---
    fig.add_trace(go.Scatter(
        x=times, y=ys, mode='markers', 
        marker=dict(color='rgba(255, 100, 100, 0.5)', size=5), 
        name='Raw Data (Y)'
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=uniform_times, y=y_filtered, mode='lines', 
        line=dict(color='green', width=2), 
        connectgaps=False,
        name='Filtered (Y)'
    ), row=1, col=2)

    # --- PLOT 3: Y VS X MAP (Row 2, Col 1 -> Colspan 2) ---
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='markers', 
        marker=dict(color='rgba(255, 100, 100, 0.5)', size=5), 
        name='Raw Data (Map)'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=x_filtered, y=y_filtered, mode='lines', 
        line=dict(color='purple', width=2), 
        connectgaps=False,
        name='Filtered (Map)'
    ), row=2, col=1)

    # --- LAYOUT SETTINGS ---
    # Inverted Y axis for map (image coordinates)
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    # Set the map X and Y axes to the same scale so the square is actually a square
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=2, col=1)
    
    # Axis labels
    fig.update_xaxes(title_text="Time (sec)", row=1, col=1)
    fig.update_yaxes(title_text="X Pixel Position", row=1, col=1)
    
    fig.update_xaxes(title_text="Time (sec)", row=1, col=2)
    fig.update_yaxes(title_text="Y Pixel Position", row=1, col=2)
    
    fig.update_xaxes(title_text="X Pixel", row=2, col=1)
    fig.update_yaxes(title_text="Y Pixel", row=2, col=1)

    # Increase height so both rows fit comfortably
    fig.update_layout(
        title_text="Trajectory Filtering Dashboard", 
        height=900, 
        width=1200, 
        hovermode="closest",
        showlegend=True
    )

    if filename:
        fig.write_html(filename)
        print(f"Interactive Plotly dashboard saved to: {filename}")
    
    if show_plot:
        fig.show()

def map_val(val, val_min, val_max, px_min, px_max):
    """ Helper function to map a data value to a pixel coordinate on the canvas. """
    if val_max == val_min:
        return int((px_min + px_max) / 2)
    ratio = (val - val_min) / (val_max - val_min)
    return int(px_min + ratio * (px_max - px_min))

def draw_custom_cv2_plot(times, xs, ys, uniform_times, x_filtered, y_filtered, filename=None, show_plot=False):
    """ Creates a 1200x600 dashboard using pure OpenCV to visualize the before/after filtering. """
    h, w = 600, 1200
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    p1_x_min, p1_x_max = 80, 550
    p1_y_min, p1_y_max = 520, 80  
    
    p2_x_min, p2_x_max = 680, 1150
    p2_y_min, p2_y_max = 80, 520  

    t_min, t_max = np.nanmin(times), np.nanmax(times)
    x_min, x_max = min(np.nanmin(xs), np.nanmin(x_filtered)), max(np.nanmax(xs), np.nanmax(x_filtered))
    y_min, y_max = min(np.nanmin(ys), np.nanmin(y_filtered)), max(np.nanmax(ys), np.nanmax(y_filtered))

    cv2.rectangle(canvas, (p1_x_min, p1_y_max), (p1_x_max, p1_y_min), (240, 240, 240), -1)
    cv2.rectangle(canvas, (p2_x_min, p2_y_min), (p2_x_max, p2_y_max), (240, 240, 240), -1)
    cv2.rectangle(canvas, (p1_x_min, p1_y_max), (p1_x_max, p1_y_min), (100, 100, 100), 2)
    cv2.rectangle(canvas, (p2_x_min, p2_y_min), (p2_x_max, p2_y_max), (100, 100, 100), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "X Coordinate over Time", (p1_x_min, 40), font, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "2D Trajectory Map (Top-Down)", (p2_x_min, 40), font, 0.7, (0, 0, 0), 2)

    for t, x in zip(times, xs):
        if not np.isnan(x):
            px = map_val(t, t_min, t_max, p1_x_min, p1_x_max)
            py = map_val(x, x_min, x_max, p1_y_min, p1_y_max)
            cv2.circle(canvas, (px, py), 2, (150, 150, 255), -1)
            
    for x, y in zip(xs, ys):
        if not np.isnan(x) and not np.isnan(y):
            px = map_val(x, x_min, x_max, p2_x_min, p2_x_max)
            py = map_val(y, y_min, y_max, p2_y_min, p2_y_max)
            cv2.circle(canvas, (px, py), 2, (150, 150, 255), -1)

    valid_mask = ~np.isnan(x_filtered)
    valid_indices = np.where(valid_mask)[0]
    
    segments = []
    for k, g in groupby(enumerate(valid_indices), lambda ix: ix[0] - ix[1]):
        segments.append(list(map(itemgetter(1), g)))
        
    for seg in segments:
        if len(seg) > 1:
            pts_p1 = []
            pts_p2 = []
            for i in seg:
                t, x, y = uniform_times[i], x_filtered[i], y_filtered[i]
                
                px1 = map_val(t, t_min, t_max, p1_x_min, p1_x_max)
                py1 = map_val(x, x_min, x_max, p1_y_min, p1_y_max)
                pts_p1.append([px1, py1])
                
                px2 = map_val(x, x_min, x_max, p2_x_min, p2_x_max)
                py2 = map_val(y, y_min, y_max, p2_y_min, p2_y_max)
                pts_p2.append([px2, py2])
                
            cv2.polylines(canvas, [np.array(pts_p1, np.int32).reshape((-1, 1, 2))], False, (255, 0, 0), 2)
            cv2.polylines(canvas, [np.array(pts_p2, np.int32).reshape((-1, 1, 2))], False, (255, 0, 0), 2)

    cv2.circle(canvas, (30, 560), 5, (150, 150, 255), -1)
    cv2.putText(canvas, "Raw Data (Spikes & Gaps)", (45, 565), font, 0.6, (0, 0, 0), 1)
    cv2.line(canvas, (20, 585), (40, 585), (255, 0, 0), 3)
    cv2.putText(canvas, "Filtered & Interpolated", (45, 590), font, 0.6, (0, 0, 0), 1)

    if filename:
        cv2.imwrite(filename, canvas)
        print(f"Dashboard saved to: {filename}")

    if show_plot:
        cv2.imshow("Trajectory Filtering Dashboard", canvas)
        print("Graph generated. Press any key on the window to close it...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ==============================================================================
# MAIN PROCESSING LOGIC
# ==============================================================================
def process_trajectory(csv_path, config_path=None, output_path=None, median_window=5, savgol_window=31, show_plot=True, use_plotly=False):
    """
    Reads a trajectory CSV, crops to a 5-minute valid window, applies GEOMETRICALLY SAFE 
    interpolation, removes outliers, applies a low-pass filter, and saves the cleaned data.
    """
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        return

    # --- 0. LOAD GEOMETRY (INNER POLYGON) ---
    inner_poly = None
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            if "inner_poly" in config_data:
                inner_poly = np.array(config_data["inner_poly"], dtype=np.int32)
                print(f"Loaded inner polygon geometry from: {config_path}")
    else:
        print("WARNING: Config file not found. Geometric wall-collision checks will be DISABLED.")

    # --- 1. LOAD RAW DATA ---
    times, xs, ys = [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader, None)  # Skip header
        for row in reader:
            try:
                times.append(float(row[0]))
                xs.append(float(row[1]))
                ys.append(float(row[2]))
            except (ValueError, IndexError):
                continue

    if len(times) < 10:
        print("ERROR: Not enough data points to process.")
        return

    times = np.array(times)
    xs = np.array(xs)
    ys = np.array(ys)
    print(f"Loaded {len(times)} raw data points.")

    # --- 2. FIND INSERTION POINT AND CROP TO 5 MINUTES ---
    t_start = detect_insertion_time(times, xs, ys)
    t_end = t_start + 300.0 + 10.0  # Exactly 5 minutes (300 seconds) + 10 seconds buffer
    # Filter arrays to keep only the valid 5-minute window
    valid_time_mask = (times >= t_start) & (times <= t_end)
    times = times[valid_time_mask]
    xs = xs[valid_time_mask]
    ys = ys[valid_time_mask]

    print(f"Rat insertion detected at T={round(t_start, 2)}s.")
    print(f"Cropped trajectory to a 5-minute window ending at T={round(times[-1], 2)}s.")
    print(f"Remaining valid points: {len(times)}")

    if len(times) < 10:
        print("ERROR: Not enough data left after cropping. Aborting.")
        return

    # --- 3. SMART INTERPOLATION ---
    time_diffs = np.diff(times)
    if len(time_diffs) == 0: return
        
    median_dt = np.median(time_diffs)
    # The uniform grid is based on the cropped start and end times!
    uniform_times = np.arange(times[0], times[-1], median_dt)
    
    x_interp = np.full_like(uniform_times, np.nan)
    y_interp = np.full_like(uniform_times, np.nan)
    
    invalid_jumps = 0
    invalid_jump_times = []

    for i in range(len(times) - 1):
        t1, t2 = times[i], times[i+1]
        p1 = (xs[i], ys[i])
        p2 = (xs[i+1], ys[i+1])
        
        mask = (uniform_times >= t1) & (uniform_times <= t2)
        
        if np.any(mask):
            time_gap = t2 - t1
            spatial_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

            if time_gap > 0.2 or spatial_dist > 50.0:
                # Geometric Check: Did the rat teleport through a wall?
                if not is_path_valid(p1, p2, inner_poly):
                    invalid_jumps += 1
                    invalid_jump_times.append((t1, t2))
                    
                    idx_t1 = np.argmin(np.abs(uniform_times - t1))
                    idx_t2 = np.argmin(np.abs(uniform_times - t2))
                    x_interp[idx_t1] = p1[0]
                    y_interp[idx_t1] = p1[1]
                    x_interp[idx_t2] = p2[0]
                    y_interp[idx_t2] = p2[1]
                    continue
            
            x_interp[mask] = np.interp(uniform_times[mask], [t1, t2], [p1[0], p2[0]])
            y_interp[mask] = np.interp(uniform_times[mask], [t1, t2], [p1[1], p2[1]])

    if invalid_jumps > 0:
        print(f"\n[!] ALERT: Detected {invalid_jumps} physically impossible jumps across the arena!")
        print("    These segments crossed the inner walls. The interpolation for these gaps was rejected.\n")

    # --- 4. SEGMENTED FILTERING ---
    x_filtered = np.copy(x_interp)
    y_filtered = np.copy(y_interp)

    valid_mask = ~np.isnan(x_interp)
    valid_indices = np.where(valid_mask)[0]
    
    segments = []
    for k, g in groupby(enumerate(valid_indices), lambda ix: ix[0] - ix[1]):
        segments.append(list(map(itemgetter(1), g)))
        
    for seg in segments:
        if len(seg) < 3:
            continue
            
        seg_x = x_interp[seg]
        seg_y = y_interp[seg]
        
        # Outlier Removal
        mw = min(median_window, len(seg))
        seg_x_med = median_filter(seg_x, size=mw)
        seg_y_med = median_filter(seg_y, size=mw)
        
        # Low-Pass Filter
        sw = min(savgol_window, len(seg))
        if sw % 2 == 0: sw -= 1
        
        if sw > 3:
            x_filtered[seg] = savgol_filter(seg_x_med, window_length=sw, polyorder=3)
            y_filtered[seg] = savgol_filter(seg_y_med, window_length=sw, polyorder=3)
        else:
            x_filtered[seg] = seg_x_med
            y_filtered[seg] = seg_y_med

    # --- 5. SAVE CLEANED DATA ---
    if output_path is None:
        filename = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join(os.path.dirname(csv_path), f"{filename}_filtered.csv")

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Time_Sec", "X", "Y"])
        
        for t, x, y in zip(uniform_times, x_filtered, y_filtered):
            # Only save those that are valid AND start after t_start
            # (Due to cropping this should be automatic, but we check for safety)
            if not np.isnan(x) and not np.isnan(y):
                writer.writerow([round(t, 3), int(np.round(x)), int(np.round(y))])

    base_dir = os.path.dirname(csv_path)
    filename_base = os.path.splitext(os.path.basename(csv_path))[0]
    
    stat_base = filename_base.replace('_trajectory', '')
    stats_file = os.path.join(base_dir, f"{stat_base}_statistics.txt")

    base_dir = os.path.dirname(csv_path)
    filename_base = os.path.splitext(os.path.basename(csv_path))[0]
    
    stat_base = filename_base.replace('_trajectory', '')
    stats_file = os.path.join(base_dir, f"{stat_base}_statistics.txt")

    if os.path.exists(stats_file):
        with open(stats_file, 'a') as f:
            f.write(f"\nFiltering Statistics (traj_filter.py)\n")
            f.write(f"--------------------------------------\n")
            f.write(f"Valid tracking window start: {round(t_start, 2)}s\n")
            f.write(f"Valid tracking window end:   {round(t_end, 2)}s\n")
            f.write(f"Physically impossible jumps rejected: {invalid_jumps}\n")
            
            if invalid_jumps > 0:
                f.write("Rejected jump time windows (sec):\n")
                for jump in invalid_jump_times:
                    f.write(f"  - from {round(jump[0], 2):>6}s  to  {round(jump[1], 2):>6}s\n")
                    
        print(f"Updated statistics file with detailed jump times: {stats_file}")
    else:
        print(f"WARNING: Could not find statistics file to update ({stats_file})")

    print(f"Success! Filtered trajectory saved to: {output_path}")

    # --- 6. VISUALIZATION ---
    base_dir = os.path.dirname(csv_path)
    filename_base = os.path.splitext(os.path.basename(csv_path))[0]
    
    if use_plotly:
        plot_path = os.path.join(base_dir, f"{filename_base}_plotly.html")
        draw_plotly_plot(times, xs, ys, uniform_times, x_filtered, y_filtered, filename=plot_path, show_plot=show_plot)
    else:
        plot_path = os.path.join(base_dir, f"{filename_base}_plot.jpg")
        draw_custom_cv2_plot(times, xs, ys, uniform_times, x_filtered, y_filtered, filename=plot_path, show_plot=show_plot)


@click.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--config", default="arena_config_poly.json", help="Path to ROI config JSON file (Used for geometry checks)")
@click.option("--output", default=None, help="Output CSV filename (optional)")
@click.option("--median-window", default=5, help="Window size for Outlier Removal (default: 5)")
@click.option("--savgol-window", default=31, help="Window size for Low-Pass Filtering (default: 31)")
@click.option("--plot", is_flag=True, help="Show the comparison graph")
@click.option("--plotly", is_flag=True, help="Use interactive Plotly graph instead of OpenCV")
def cli(csv_path, config, output, median_window, savgol_window, plot, plotly):
    """
    Interpolates missing frames, applies geometric wall-collision checks, 
    crops data to the first 5 minutes of activity, and removes tracking outliers.
    """
    process_trajectory(
        csv_path, 
        config_path=config,
        output_path=output, 
        median_window=median_window, 
        savgol_window=savgol_window, 
        show_plot=plot,
        use_plotly=plotly
    )

if __name__ == '__main__':
    cli()