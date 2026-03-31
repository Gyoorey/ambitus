import os
import csv
import json
import numpy as np
from pathlib import Path
import click

def calibrate_global_thresholds(results_dir, output_json="global_thresholds.json"):
    """
    Iterates through all *_filtered.csv files in the specified directory (results_dir).
    Calculates the aggregated population speed distribution and determines the
    global T_lower and T_upper thresholds. Saves the result to a JSON file.
    """
    root_dir = Path(results_dir).resolve()
    
    if not root_dir.exists():
        print(f"ERROR: Results directory not found: {root_dir}")
        return False

    csv_files = list(root_dir.rglob('*_trajectory_filtered.csv'))
    if not csv_files:
        # Fallback, if no filtered files found, try plain trajectory files
        csv_files = list(root_dir.rglob('*_trajectory.csv'))
        
    if not csv_files:
        print(f"ERROR: No trajectory CSV files found in {root_dir}")
        return False
        
    print(f"Calibrating global thresholds using {len(csv_files)} files...")
    
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
        
        # Calculate instantaneous speeds (dt, dx, dy)
        dt = np.diff(times)
        dt[dt == 0] = 0.001
        dx = np.diff(xs)
        dy = np.diff(ys)
        
        distances = np.sqrt(dx**2 + dy**2)
        speeds = distances / dt
        
        # Only collect real movements (filter out noise/zeros)
        moving_speeds = speeds[speeds > 0.1]
        all_speeds.extend(moving_speeds.tolist())
        
    if not all_speeds:
        print("ERROR: No movement detected in any of the files.")
        return False
        
    # --- CALCULATE GLOBAL THRESHOLDS ---
    all_speeds = np.array(all_speeds)
    
    t_lower = np.percentile(all_speeds, 15)
    t_upper = np.percentile(all_speeds, 40)
    
    thresholds = {
        "T_lower_px_s": round(float(t_lower), 2),
        "T_upper_px_s": round(float(t_upper), 2),
        "Population_Size_Files": len(csv_files),
        "Total_Data_Points": len(all_speeds)
    }
    
    output_path = root_dir / output_json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, indent=4)
        
    print(f"Calibration Complete!")
    print(f"  - Files analyzed: {len(csv_files)}")
    print(f"  - Total movement data points: {len(all_speeds)}")
    print(f"  - Global T_lower (15th percentile): {thresholds['T_lower_px_s']} px/s")
    print(f"  - Global T_upper (40th percentile): {thresholds['T_upper_px_s']} px/s")
    print(f"Saved to: {output_path}")
    
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