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
# VISUALIZATION (PLOTLY & CV2)
# ==============================================================================
def draw_plotly_plot(times, xs, ys, uniform_times, x_filtered, y_filtered, filename=None, show_plot=False):
    """
    Creates an interactive 1200x600 dashboard using Plotly.
    Saves it as an HTML file and opens it in the default browser.
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=("X Coordinate over Time", "2D Trajectory Map (Top-Down)"))

    # --- PLOT 1: X OVER TIME ---
    fig.add_trace(go.Scatter(
        x=times, y=xs, mode='markers', 
        marker=dict(color='rgba(255, 100, 100, 0.5)', size=5), 
        name='Raw Data (X)'
    ), row=1, col=1)
    
    # Notice we use 'connectgaps=False' so the Plotly line breaks where NaN values exist!
    fig.add_trace(go.Scatter(
        x=uniform_times, y=x_filtered, mode='lines', 
        line=dict(color='blue', width=2), 
        connectgaps=False,
        name='Filtered & Interpolated (X)'
    ), row=1, col=1)

    # --- PLOT 2: Y VS X MAP ---
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='markers', 
        marker=dict(color='rgba(255, 100, 100, 0.5)', size=5), 
        name='Raw Data (Map)'
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=x_filtered, y=y_filtered, mode='lines', 
        line=dict(color='blue', width=2), 
        connectgaps=False,
        name='Filtered & Interpolated (Map)'
    ), row=1, col=2)

    # --- LAYOUT SETTINGS ---
    # Inverse Y axis on the 2D map to match OpenCV/Image coordinates
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    
    fig.update_xaxes(title_text="Time (sec)", row=1, col=1)
    fig.update_yaxes(title_text="X Pixel Position", row=1, col=1)
    
    fig.update_xaxes(title_text="X Pixel", row=1, col=2)
    fig.update_yaxes(title_text="Y Pixel", row=1, col=2)

    fig.update_layout(title_text="Trajectory Filtering Dashboard", height=600, width=1200, hovermode="closest")

    if filename:
        fig.write_html(filename)
        print(f"Interactive Plotly dashboard saved to: {filename}")
    
    # Opens in the default browser
    if show_plot:
        fig.show()

def map_val(val, val_min, val_max, px_min, px_max):
    """ Helper function to map a data value to a pixel coordinate on the canvas. """
    if val_max == val_min:
        return int((px_min + px_max) / 2)
    ratio = (val - val_min) / (val_max - val_min)
    return int(px_min + ratio * (px_max - px_min))

def draw_custom_cv2_plot(times, xs, ys, uniform_times, x_filtered, y_filtered, filename=None, show_plot=False):
    """
    Creates a 1200x600 dashboard using pure OpenCV to visualize the before/after filtering.
    """
    h, w = 600, 1200
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    p1_x_min, p1_x_max = 80, 550
    p1_y_min, p1_y_max = 520, 80  
    
    p2_x_min, p2_x_max = 680, 1150
    p2_y_min, p2_y_max = 80, 520  

    t_min, t_max = np.nanmin(times), np.nanmax(times)
    # Use nanmin/nanmax because our filtered arrays now contain NaNs!
    x_min, x_max = min(np.nanmin(xs), np.nanmin(x_filtered)), max(np.nanmax(xs), np.nanmax(x_filtered))
    y_min, y_max = min(np.nanmin(ys), np.nanmin(y_filtered)), max(np.nanmax(ys), np.nanmax(y_filtered))

    cv2.rectangle(canvas, (p1_x_min, p1_y_max), (p1_x_max, p1_y_min), (240, 240, 240), -1)
    cv2.rectangle(canvas, (p2_x_min, p2_y_min), (p2_x_max, p2_y_max), (240, 240, 240), -1)
    cv2.rectangle(canvas, (p1_x_min, p1_y_max), (p1_x_max, p1_y_min), (100, 100, 100), 2)
    cv2.rectangle(canvas, (p2_x_min, p2_y_min), (p2_x_max, p2_y_max), (100, 100, 100), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "X Coordinate over Time", (p1_x_min, 40), font, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "2D Trajectory Map (Top-Down)", (p2_x_min, 40), font, 0.7, (0, 0, 0), 2)

    # Draw raw points (Light Red)
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

    # Draw filtered lines (Blue) - We must draw them segment by segment to avoid connecting across NaNs
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

    # Add Legends
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
    Reads a trajectory CSV, applies GEOMETRICALLY SAFE interpolation for missing frames, 
    removes outliers, applies a low-pass filter, and saves the cleaned integer data.
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

    # --- 2. SMART INTERPOLATION ---
    time_diffs = np.diff(times)
    if len(time_diffs) == 0: return
        
    median_dt = np.median(time_diffs)
    uniform_times = np.arange(times[0], times[-1], median_dt)
    
    # Initialize arrays with NaNs (gaps)
    x_interp = np.full_like(uniform_times, np.nan)
    y_interp = np.full_like(uniform_times, np.nan)
    
    invalid_jumps = 0

    # Process segment by segment to respect the geometry
    for i in range(len(times) - 1):
        t1, t2 = times[i], times[i+1]
        p1 = (xs[i], ys[i])
        p2 = (xs[i+1], ys[i+1])
        
        # Find which uniform time steps fall into this gap
        mask = (uniform_times >= t1) & (uniform_times <= t2)
        
        if np.any(mask):
            # Geometric Check: Did the rat teleport through a wall?
            if not is_path_valid(p1, p2, inner_poly):
                invalid_jumps += 1
                
                # Keep the measured endpoints, but leave the path between them as NaN (Broken line)git 
                idx_t1 = np.argmin(np.abs(uniform_times - t1))
                idx_t2 = np.argmin(np.abs(uniform_times - t2))
                x_interp[idx_t1] = p1[0]
                y_interp[idx_t1] = p1[1]
                x_interp[idx_t2] = p2[0]
                y_interp[idx_t2] = p2[1]
                continue
            
            # If path is valid, linearly interpolate the gap
            x_interp[mask] = np.interp(uniform_times[mask], [t1, t2], [p1[0], p2[0]])
            y_interp[mask] = np.interp(uniform_times[mask], [t1, t2], [p1[1], p2[1]])

    if invalid_jumps > 0:
        print(f"\n[!] ALERT: Detected {invalid_jumps} physically impossible jumps across the arena!")
        print("    These segments crossed the inner walls. The interpolation for these gaps was rejected.\n")

    # --- 3. SEGMENTED FILTERING ---
    # Because we now have intentional NaNs in the data, we must filter continuous blocks separately.
    x_filtered = np.copy(x_interp)
    y_filtered = np.copy(y_interp)

    valid_mask = ~np.isnan(x_interp)
    valid_indices = np.where(valid_mask)[0]
    
    segments = []
    for k, g in groupby(enumerate(valid_indices), lambda ix: ix[0] - ix[1]):
        segments.append(list(map(itemgetter(1), g)))
        
    for seg in segments:
        if len(seg) < 3:
            continue # Too short to filter, keep raw interpolated values
            
        seg_x = x_interp[seg]
        seg_y = y_interp[seg]
        
        # Outlier Removal (Median)
        mw = min(median_window, len(seg))
        seg_x_med = median_filter(seg_x, size=mw)
        seg_y_med = median_filter(seg_y, size=mw)
        
        # Low-Pass Filter (Savitzky-Golay)
        sw = min(savgol_window, len(seg))
        if sw % 2 == 0: sw -= 1
        
        if sw > 3:
            x_filtered[seg] = savgol_filter(seg_x_med, window_length=sw, polyorder=3)
            y_filtered[seg] = savgol_filter(seg_y_med, window_length=sw, polyorder=3)
        else:
            x_filtered[seg] = seg_x_med
            y_filtered[seg] = seg_y_med

    # --- 4. SAVE CLEANED DATA ---
    if output_path is None:
        filename = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join(os.path.dirname(csv_path), f"{filename}_filtered.csv")

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Time_Sec", "X", "Y"])
        
        for t, x, y in zip(uniform_times, x_filtered, y_filtered):
            # Only save valid (non-NaN) points. The missing jumps will remain missing in the CSV.
            if not np.isnan(x) and not np.isnan(y):
                # Enforce STRICT integers for pixel coordinates
                writer.writerow([round(t, 3), int(np.round(x)), int(np.round(y))])
            
    print(f"Success! Filtered trajectory saved to: {output_path}")

    # --- 5. VISUALIZATION ---
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
    Interpolates missing frames (respecting arena geometry), removes tracking outliers, 
    and applies a low-pass filter to the trajectory CSV.
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