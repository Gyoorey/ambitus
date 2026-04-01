import os
import csv
import json
import numpy as np
from pathlib import Path
import click
import plotly.graph_objects as go

def calibrate_global_thresholds(results_dir, output_json="global_thresholds.json"):
    """
    Iterates through all *_filtered.csv files in the specified directory (results_dir).
    Calculates the aggregated population speed distribution and determines the
    global T_lower and T_upper thresholds. Saves the result to a JSON file and
    generates a Plotly histogram for visual verification.
    """
    root_dir = Path(results_dir).resolve()
    
    if not root_dir.exists():
        print(f"ERROR: Results directory not found: {root_dir}")
        return False

    csv_files = list(root_dir.rglob('*_trajectory_filtered.csv'))
    if not csv_files:
        print(f"No filtered trajectory CSV files found in {root_dir}. Trying unfiltered trajectory files...")
        # Fallback, if no filtered files found, try plain trajectory files
        csv_files = list(root_dir.rglob('*_trajectory.csv'))
        
    if not csv_files:
        print(f"ERROR: No trajectory CSV files found in {root_dir}")
        return False
        
    print(f"Calibrating global thresholds using {len(csv_files)} files...")
    
    all_moving_speeds = []
    all_speeds = []
    
    for csv_file in csv_files:
        times, xs, ys = [], [], []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader, None)
            for row in reader:
                try:
                    times.append(float(row[0]))
                    xs.append(float(row[1]))
                    ys.append(float(row[2]))
                except (ValueError, IndexError):
                    continue
                    
        if len(times) < 10:
            continue
            
        times, xs, ys = np.array(times), np.array(xs), np.array(ys)
        
        # Calculate speeds (dt, dx, dy)
        dt = np.diff(times)
        dx = np.diff(xs)
        dy = np.diff(ys)
        
        distances = np.sqrt(dx**2 + dy**2)
        speeds = distances / dt

        total_time = times[-1] - times[0]
        avg_dt = total_time / len(times) if len(times) > 0 else 0.04        
        noise_floor_speed = 1.0 / avg_dt if avg_dt > 0 else 25.0
        
        if len(times) == 0:
            print(f"WARNING: No valid time data in {csv_file}.")
            continue
            
        # Only collect real movements (filter out noise/zeros)
        moving_speeds = speeds[speeds > noise_floor_speed]
        all_moving_speeds.extend(moving_speeds.tolist())
        all_speeds.extend(speeds.tolist())
        
    if not all_moving_speeds:
        print("ERROR: No movement detected in any of the files.")
        return False
        
    # --- CALCULATE GLOBAL THRESHOLDS ---
    all_moving_speeds = np.array(all_moving_speeds)
    
    t_lower = np.percentile(all_moving_speeds, 15)
    t_upper = np.percentile(all_moving_speeds, 40)
    
    thresholds = {
        "T_lower_px_s": round(float(t_lower), 2),
        "T_upper_px_s": round(float(t_upper), 2),
        "Population_Size_Files": len(csv_files),
        "Total_Moving_Samples": len(all_moving_speeds),
        "Total_Data_Samples": len(all_speeds),
    }
    
    # 1. Save JSON
    output_path = root_dir / output_json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, indent=4)
        
    # --- GENERATE PLOTLY DISTRIBUTION GRAPH ---
    # To prevent huge HTML files and browser freezing, we downsample if we have too many points
    plot_data = all_moving_speeds
    max_plot_samples = 100000
    if len(plot_data) > max_plot_samples:
        plot_data = np.random.choice(plot_data, size=max_plot_samples, replace=False)

    fig = go.Figure()

    # Create Histogram of moving speeds
    fig.add_trace(go.Histogram(
        x=plot_data,
        nbinsx=300,
        name="Moving Speeds",
        marker_color="royalblue",
        opacity=0.75
    ))

    # Add Vertical lines for the calculated thresholds
    fig.add_vline(x=t_lower, line_dash="dash", line_color="red", line_width=2,
                  annotation_text=f"T_lower (15%): {t_lower:.1f} px/s", annotation_position="top right")
    fig.add_vline(x=t_upper, line_dash="dash", line_color="green", line_width=2,
                  annotation_text=f"T_upper (40%): {t_upper:.1f} px/s", annotation_position="top right")

    # Layout adjustments: We crop the X axis at the 98th percentile because 
    # extreme outlier sprint speeds can stretch the graph, making the threshold area invisible.
    x_max_display = np.percentile(plot_data, 98)

    fig.update_layout(
        title=f"Global Population Speed Distribution (Analyzed {len(csv_files)} files)",
        xaxis_title="Speed (px/s)",
        yaxis_title="Sample Count",
        xaxis=dict(range=[0, x_max_display]),
        bargap=0.05,
        template="plotly_white"
    )

    # Save HTML (Do NOT show it)
    html_filename = output_json.replace('.json', '_distribution.html')
    html_output_path = root_dir / html_filename
    fig.write_html(str(html_output_path))
    
    print(f"Calibration Complete!")
    print(f"  - Files analyzed: {len(csv_files)}")
    print(f"  - Total movement data points: {len(all_moving_speeds)}")
    print(f"  - Total data points: {len(all_speeds)}")
    print(f"  - Global T_lower (15th percentile): {thresholds['T_lower_px_s']} px/s")
    print(f"  - Global T_upper (40th percentile): {thresholds['T_upper_px_s']} px/s")
    print(f"Saved JSON to: {output_path}")
    print(f"Saved Plotly Graph to: {html_output_path}")
    
    return True

@click.command()
@click.argument("results_dir", type=click.Path())
@click.option("--output", default="global_thresholds.json", help="Output JSON filename")
def cli(results_dir, output):
    """
    Calculates population-level global speed thresholds for Active/Inactive phase classification.
    """
    calibrate_global_thresholds(results_dir, output)

if __name__ == '__main__':
    cli()