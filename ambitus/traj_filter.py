import csv
import os
import click
import numpy as np
import cv2
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    
    fig.add_trace(go.Scatter(
        x=uniform_times, y=x_filtered, mode='lines', 
        line=dict(color='blue', width=2), 
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
    # Prevent division by zero if all values are identical
    if val_max == val_min:
        return int((px_min + px_max) / 2)
    
    # Linear interpolation mapped to pixels
    ratio = (val - val_min) / (val_max - val_min)
    return int(px_min + ratio * (px_max - px_min))

def draw_custom_cv2_plot(times, xs, ys, uniform_times, x_filtered, y_filtered, filename=None, show_plot=False):
    """
    Creates a 1200x600 dashboard using pure OpenCV to visualize the before/after filtering.
    Left Panel: X-Coordinate over Time.
    Right Panel: 2D Trajectory Map (X vs Y).
    """
    # Create a white canvas (Height, Width, Channels)
    h, w = 600, 1200
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # --- Define Plot Boundaries (Margins) ---
    # Plot 1 (Left): X vs Time
    p1_x_min, p1_x_max = 80, 550
    p1_y_min, p1_y_max = 520, 80  # Inverted: smaller pixel value = higher up
    
    # Plot 2 (Right): Y vs X (Top-Down Map)
    p2_x_min, p2_x_max = 680, 1150
    p2_y_min, p2_y_max = 80, 520  # Direct mapping (Y goes down in images)

    # Calculate data bounds
    t_min, t_max = np.min(times), np.max(times)
    x_min, x_max = min(np.min(xs), np.min(x_filtered)), max(np.max(xs), np.max(x_filtered))
    y_min, y_max = min(np.min(ys), np.min(y_filtered)), max(np.max(ys), np.max(y_filtered))

    # --- DRAW BACKGROUND AND AXES ---
    cv2.rectangle(canvas, (p1_x_min, p1_y_max), (p1_x_max, p1_y_min), (240, 240, 240), -1)
    cv2.rectangle(canvas, (p2_x_min, p2_y_min), (p2_x_max, p2_y_max), (240, 240, 240), -1)
    
    cv2.rectangle(canvas, (p1_x_min, p1_y_max), (p1_x_max, p1_y_min), (100, 100, 100), 2)
    cv2.rectangle(canvas, (p2_x_min, p2_y_min), (p2_x_max, p2_y_max), (100, 100, 100), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "X Coordinate over Time", (p1_x_min, 40), font, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "2D Trajectory Map (Top-Down)", (p2_x_min, 40), font, 0.7, (0, 0, 0), 2)

    # --- DRAW PLOT 1: X OVER TIME ---
    # Draw raw points (Light Red)
    for t, x in zip(times, xs):
        px = map_val(t, t_min, t_max, p1_x_min, p1_x_max)
        py = map_val(x, x_min, x_max, p1_y_min, p1_y_max)
        cv2.circle(canvas, (px, py), 2, (150, 150, 255), -1)  # BGR: Light Red
        
    # Draw filtered line (Blue)
    pts_p1 = []
    for t, x in zip(uniform_times, x_filtered):
        px = map_val(t, t_min, t_max, p1_x_min, p1_x_max)
        py = map_val(x, x_min, x_max, p1_y_min, p1_y_max)
        pts_p1.append([px, py])
    
    pts_p1 = np.array(pts_p1, np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts_p1], False, (255, 0, 0), 2)  # BGR: Blue

    # --- DRAW PLOT 2: Y VS X MAP ---
    # Draw raw points (Light Red)
    for x, y in zip(xs, ys):
        px = map_val(x, x_min, x_max, p2_x_min, p2_x_max)
        py = map_val(y, y_min, y_max, p2_y_min, p2_y_max)
        cv2.circle(canvas, (px, py), 2, (150, 150, 255), -1)

    # Draw filtered line (Blue)
    pts_p2 = []
    for x, y in zip(x_filtered, y_filtered):
        px = map_val(x, x_min, x_max, p2_x_min, p2_x_max)
        py = map_val(y, y_min, y_max, p2_y_min, p2_y_max)
        pts_p2.append([px, py])
        
    pts_p2 = np.array(pts_p2, np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts_p2], False, (255, 0, 0), 2)

    # --- ADD LEGENDS ---
    cv2.circle(canvas, (30, 560), 5, (150, 150, 255), -1)
    cv2.putText(canvas, "Raw Data (Spikes & Gaps)", (45, 565), font, 0.6, (0, 0, 0), 1)
    
    cv2.line(canvas, (20, 585), (40, 585), (255, 0, 0), 3)
    cv2.putText(canvas, "Filtered & Interpolated", (45, 590), font, 0.6, (0, 0, 0), 1)

    if filename:
        cv2.imwrite(filename, canvas)
        print(f"Dashboard saved to: {filename}")

    if show_plot:
        # Show the dashboard
        cv2.imshow("Trajectory Filtering Dashboard", canvas)
        print("Graph generated. Press any key on the window to close it...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_trajectory(csv_path, output_path=None, median_window=5, savgol_window=31, show_plot=True, use_plotly=False):
    """
    Reads a trajectory CSV, interpolates missing frames, removes outliers, 
    applies a low-pass filter, and saves the cleaned data.
    """
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        return

    # 1. Read the raw data
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

    # Convert to numpy arrays for vector math
    times = np.array(times)
    xs = np.array(xs)
    ys = np.array(ys)

    print(f"Loaded {len(times)} raw data points.")

    # --- STEP 1: INTERPOLATION (Fill missing frames) ---
    # Calculate the median time difference (dt) to find the original frame rate
    time_diffs = np.diff(times)
    if len(time_diffs) == 0:
        return
        
    median_dt = np.median(time_diffs)
    
    # Create a uniform time grid from start to end, strictly spaced by median_dt
    uniform_times = np.arange(times[0], times[-1], median_dt)
    
    # Linearly interpolate X and Y coordinates for the missing timestamps
    x_interp = np.interp(uniform_times, times, xs)
    y_interp = np.interp(uniform_times, times, ys)
    
    print(f"Interpolated to {len(uniform_times)} points (filled gaps).")

    # --- STEP 2: OUTLIER REMOVAL (Median Filter) ---
    # The median filter removes sudden spikes (e.g., 1-frame tracking glitches)
    # without blurring the actual edges/turns of the trajectory.
    x_med = median_filter(x_interp, size=median_window)
    y_med = median_filter(y_interp, size=median_window)

    # --- STEP 3: LOW-PASS FILTER (Savitzky-Golay) ---
    # The Savitzky-Golay filter removes high-frequency noise (jitter) 
    # while preserving the shape and curves of the rat's path.
    
    # Ensure the window size is odd (required by savgol_filter)
    if savgol_window % 2 == 0:
        savgol_window += 1
        
    # Ensure window is not larger than the dataset
    savgol_window = min(savgol_window, len(x_med) - 1)
    if savgol_window % 2 == 0: savgol_window -= 1

    poly_order = 3  # 3rd degree polynomial fits the curves nicely
    if savgol_window > poly_order:
        x_filtered = savgol_filter(x_med, window_length=savgol_window, polyorder=poly_order)
        y_filtered = savgol_filter(y_med, window_length=savgol_window, polyorder=poly_order)
    else:
        x_filtered, y_filtered = x_med, y_med  # Fallback if data is extremely short

    # Convert the final filtered coordinates strictly to integers
    x_filtered = np.round(x_filtered).astype(int)
    y_filtered = np.round(y_filtered).astype(int)

    # --- STEP 4: SAVE THE CLEANED DATA ---
    if output_path is None:
        filename = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join(os.path.dirname(csv_path), f"{filename}_filtered.csv")

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Time_Sec", "X", "Y"])
        # Round coordinates to 1 decimal place for neatness, but keep float precision
        for t, x, y in zip(uniform_times, x_filtered, y_filtered):
            writer.writerow([round(t, 3), round(x, 1), round(y, 1)])
            
    print(f"Success! Filtered trajectory saved to: {output_path}")

    # --- STEP 5: VISUALIZATION (Comparison) ---
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
@click.option("--output", default=None, help="Output CSV filename (optional)")
@click.option("--median-window", default=5, help="Window size for Outlier Removal (default: 5)")
@click.option("--savgol-window", default=31, help="Window size for Low-Pass Filtering (default: 31)")
@click.option("--plot", is_flag=True, help="Show the comparison graph")
@click.option("--plotly", is_flag=True, help="Use interactive Plotly graph instead of OpenCV")
def cli(csv_path, output, median_window, savgol_window, plot, plotly):
    """
    Interpolates missing frames, removes tracking outliers, and applies a low-pass filter to the trajectory CSV.
    """
    process_trajectory(
        csv_path, 
        output_path=output, 
        median_window=median_window, 
        savgol_window=savgol_window, 
        show_plot=plot,
        use_plotly=plotly
    )

if __name__ == '__main__':
    cli()