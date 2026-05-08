import os
import json
import click
from pathlib import Path

# --- THRESHOLDS FOR DETECTING PROBLEMATIC VIDEOS ---
MIN_EVALUATED_TIME_S = 280.0       # Minimum valid time (seconds) to be considered a full video
MAX_ALLOWED_JUMPS = 5              # Max allowed physically impossible jumps before flagging
MAX_MISSED_FRAMES_RATIO = 0.15     # If missed frames > 15% of total frames, flag as escaped/lost
EARLY_START_THRESHOLD_S = 20.0     # If tracking starts before this time, flag as possible false-start artifact

def parse_statistics_file(stat_path):
    """ 
    Parses the _statistics.txt file to extract missed frames, jump counts, and window start times. 
    Supports both legacy and new tracking log formats.
    """
    stats = {
        "Total_Frames": 0,
        "Missed_Frames": 0,
        "Impossible_Jumps": 0,  
        "Spikes_Removed": 0,     
        "Window_Start": None     # New feature for early start detection
    }
    
    if not os.path.exists(stat_path):
        return stats
        
    with open(stat_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # General Tracking Stats
            if line.startswith("Total Frames Processed:"):
                try: stats["Total_Frames"] = int(line.split(":")[1].strip())
                except: pass
            elif line.startswith("Missed Frames:"):
                try: stats["Missed_Frames"] = int(line.split(":")[1].strip())
                except: pass
                
            # Legacy and New Filter Stats
            elif line.startswith("Physically impossible jumps rejected:"): 
                try: stats["Impossible_Jumps"] += int(line.split(":")[1].strip())
                except: pass
            elif line.startswith("Physically impossible gaps rejected:"): 
                try: stats["Impossible_Jumps"] += int(line.split(":")[1].strip())
                except: pass
            elif line.startswith("Teleportation spikes removed:"): 
                try: stats["Spikes_Removed"] = int(line.split(":")[1].strip())
                except: pass
                
            # Window Timing Stats
            elif line.startswith("Valid tracking window start:"):
                try: 
                    # Format: "Valid tracking window start: 30.5s" -> Extracts 30.5
                    time_str = line.split(":")[1].strip().replace("s", "")
                    stats["Window_Start"] = float(time_str)
                except: pass
                
    return stats

def audit_results(results_dir, output_filename="Problematic_Videos_Report.txt"):
    root_dir = Path(results_dir).resolve()
    
    if not root_dir.exists():
        print(f"ERROR: Results directory not found: {root_dir}")
        return

    # Find all feature JSON files (indicates a processed video)
    json_files = list(root_dir.rglob('*_features.json'))
    
    if not json_files:
        print(f"No feature JSON files found in {root_dir}. Run extraction first.")
        return

    problematic_videos = []
    total_scanned = 0

    print("Scanning videos for anomalies...")

    for jpath in json_files:
        total_scanned += 1
        issues = []
        
        # 1. Determine Relative Path (e.g., Experiment_1/Video_01)
        rel_path = jpath.relative_to(root_dir)
        video_folder = rel_path.parent.name
        group_folder = str(rel_path.parent.parent) if str(rel_path.parent.parent) != '.' else 'Root'
        identifier = f"{group_folder}/{video_folder}"

        # 2. Check JSON for Total Time
        try:
            with open(jpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                eval_time = data.get("Metadata", {}).get("Total_Time_Evaluated_s", 0)
                
                if eval_time < MIN_EVALUATED_TIME_S:
                    issues.append(f"[TOO SHORT] Video only has {eval_time}s of valid evaluated time (Limit: {MIN_EVALUATED_TIME_S}s).")
        except Exception as e:
            issues.append(f"[CORRUPT JSON] Could not read features.json: {e}")

        # 3. Check Statistics TXT for Jumps, Escapes, and Early Starts
        base_name = jpath.stem.replace('_features', '')
        stat_path = jpath.parent / f"{base_name}_statistics.txt"
        
        stats = parse_statistics_file(stat_path)
        
        if stats["Impossible_Jumps"] > MAX_ALLOWED_JUMPS:
            msg = f"[TRACKING ERROR] Excessive wall jumping/gaps detected! ({stats['Impossible_Jumps']} gaps)."
            if stats["Spikes_Removed"] > 0:
                msg += f" (Also removed {stats['Spikes_Removed']} teleportation spikes)."
            issues.append(msg)
            
        if stats["Total_Frames"] > 0:
            missed_ratio = stats["Missed_Frames"] / stats["Total_Frames"]
            if missed_ratio > MAX_MISSED_FRAMES_RATIO:
                issues.append(f"[ESCAPED/LOST] Rat was lost for {round(missed_ratio*100, 1)}% of the video ({stats['Missed_Frames']} missed frames). It likely climbed out of the inner arena.")

        # --- NEW RULE: Early Start Detection ---
        if stats["Window_Start"] is not None and stats["Window_Start"] < EARLY_START_THRESHOLD_S:
            issues.append(f"[EARLY START] Analysis started suspiciously early at {stats['Window_Start']}s. (Threshold: {EARLY_START_THRESHOLD_S}s). Possible hand/shadow artifact detected as rat.")

        # If issues found, record them
        if issues:
            problematic_videos.append({
                "identifier": identifier,
                "issues": issues
            })

    # --- WRITE REPORT ---
    report_path = root_dir / output_filename
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("AMBITUS PROJECT - AUDIT REPORT\n")
        f.write("==============================\n")
        f.write(f"Total Videos Scanned: {total_scanned}\n")
        f.write(f"Problematic Videos Found: {len(problematic_videos)}\n")
        f.write("==============================\n\n")
        
        if not problematic_videos:
            f.write("No problematic videos found. All videos meet the quality thresholds!\n")
        else:
            for idx, pv in enumerate(problematic_videos, 1):
                f.write(f"{idx}. {pv['identifier']}\n")
                for issue in pv['issues']:
                    f.write(f"   - {issue}\n")
                f.write("\n")

    print(f"\nAudit complete! Found {len(problematic_videos)} problematic videos.")
    print(f"Report generated at: {report_path}")

@click.command()
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--output", default="Problematic_Videos_Report.txt", help="Name of the output text report file.")
def cli(results_dir, output):
    """
    Scans processed results for problematic videos (rat climbed out, video too short, tracking errors/wall jumps).
    Generates a human-readable text report in the results directory.
    """
    audit_results(results_dir, output_filename=output)

if __name__ == '__main__':
    cli()