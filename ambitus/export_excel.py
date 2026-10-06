import os
import re
import json
import click
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("ERROR: 'pandas' or 'openpyxl' is not installed.")
    print("Please run: pip install pandas openpyxl")
    exit(1)

def flatten_dict_smart(d, parent_key=''):
    """
    Recursively flattens the JSON dictionary.
    NEW LOGIC: Only keeps the names of 'Minute_X' and 'Metadata' groups as prefixes,
    since keys repeat within these. Completely discards the names of other main groups (I_..., II_...).
    """
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            # Only minutes and metadata get a prefix
            if k.startswith("Minute_") or k == "Metadata":
                new_parent = f"{k}_"
            else:
                new_parent = ""  # Discard the category name
                
            items.extend(flatten_dict_smart(v, new_parent).items())
        else:
            final_key = f"{parent_key}{k}"
            items.append((final_key, v))
    return dict(items)

# [MOD] Animal information columns, filled from the video and folder names.
# Video name:  <LE|LI>_G<generation>_N<rat ID>_D<day>_T<trial>   e.g. LI_G13_N30_D1_T4
# Folders:     <Control|Lisket>/<Alc|No_Alc>/ID<rat ID>/D<day>/<video>
ANIMAL_COLS = ['Gen', 'Number', 'NR', 'GR', 'Treatment']
VIDEO_NAME_RE = re.compile(r'^(LE|LI)_G(\d+)_N(\d+)_D(\d+)_T(\d+)', re.IGNORECASE)
GR_FROM_PREFIX = {'LE': 'LE', 'LI': 'Lisket'}
GR_FROM_FOLDER = {'control': 'LE', 'lisket': 'Lisket'}
TREATMENT_FROM_FOLDER = {'alc': 'Alcohol', 'no_alc': 'NO'}


def parse_animal_info(video_folder, group_folder):
    """Gen (generation), Number (rat ID), NR (left empty), GR (LE / Lisket), Treatment (NO / Alcohol).
    Gen, Number and GR come from the video name; the folder names are used when the video name does not
    match the pattern, and are cross-checked otherwise."""
    info = {'Gen': None, 'Number': None, 'NR': None, 'GR': None, 'Treatment': None}
    parts = [p for p in Path(group_folder).parts if p not in ('Root', '.')]
    lower = [p.lower() for p in parts]

    m = VIDEO_NAME_RE.match(video_folder)
    if m:
        info['GR'] = GR_FROM_PREFIX[m.group(1).upper()]
        info['Gen'] = int(m.group(2))
        info['Number'] = int(m.group(3))

    # Fallback / cross-check from the folders
    folder_gr = next((GR_FROM_FOLDER[p] for p in lower if p in GR_FROM_FOLDER), None)
    folder_id = next((int(p[2:]) for p in lower if re.fullmatch(r'id\d+', p)), None)
    if info['GR'] is None:
        info['GR'] = folder_gr
    elif folder_gr and folder_gr != info['GR']:
        print(f"WARNING: {video_folder}: group from video name ({info['GR']}) differs from folder ({folder_gr})")
    if info['Number'] is None:
        info['Number'] = folder_id
    elif folder_id is not None and folder_id != info['Number']:
        print(f"WARNING: {video_folder}: rat ID from video name ({info['Number']}) differs from folder (ID{folder_id})")

    info['Treatment'] = next((TREATMENT_FROM_FOLDER[p] for p in lower if p in TREATMENT_FROM_FOLDER), None)
    if info['Treatment'] is None or info['Gen'] is None:
        print(f"WARNING: {group_folder}/{video_folder}: could not determine "
              f"{', '.join(k for k in ('Gen', 'Treatment') if info[k] is None)}")
    return info


def export_features_to_excel(results_dir, create_folder_excels=False, target_video=None, target_group=None):
    root_dir = Path(results_dir).resolve()
    
    if not root_dir.exists():
        print(f"ERROR: Results directory not found: {root_dir}")
        return

    # Find all feature JSON files
    json_files = list(root_dir.rglob('*_features.json'))
    
    if not json_files:
        print(f"No feature JSON files found in {root_dir} or its subdirectories.")
        return
        
    print(f"Found {len(json_files)} feature JSON files. Processing...")

    all_data = []

    for jpath in json_files:
        with open(jpath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"WARNING: Could not parse JSON in file {jpath.name}")
                continue
        
        # Clean, smart flattening
        flat_data = flatten_dict_smart(data)
        
        # Extract folder information
        rel_path = jpath.relative_to(root_dir)
        parent_dir = rel_path.parent
        
        video_folder = parent_dir.name
        group_folder = str(parent_dir.parent) if str(parent_dir.parent) != '.' else 'Root'
        
        flat_data['Group_Folder'] = group_folder
        flat_data['Video_Folder'] = video_folder
        flat_data.update(parse_animal_info(video_folder, group_folder))  # [MOD]

        # Store the original file path
        flat_data['__Source_Dir'] = str(jpath.parent)
        
        all_data.append(flat_data)

    df = pd.DataFrame(all_data)
    for c in ('Gen', 'Number'):  # [MOD] whole numbers, empty cell if unknown
        df[c] = df[c].astype('Int64')

    # Set column order
    cols = df.columns.tolist()
    front_cols = ANIMAL_COLS + ['Group_Folder', 'Video_Folder', 'Metadata_Analyzed_File']  # [MOD]
    existing_front_cols = [c for c in front_cols if c in cols]
    
    # Filter out helper columns and already positioned columns
    rest_cols = [c for c in cols if c not in existing_front_cols and c != '__Source_Dir']
    
    # Final display column list
    export_cols = existing_front_cols + rest_cols
    
    # --- 1. ALWAYS UPDATE MASTER EXCEL ---
    master_out = root_dir / "Ambitus_Master_Features.xlsx"
    df[export_cols].to_excel(master_out, index=False)
    print(f"SUCCESS: Master Excel saved/updated at {master_out}")

    # --- 2. TARGETED OR FULL EXCEL GENERATION ---
    if create_folder_excels or target_video:
        print("\nCreating/Updating individual folder Excels...")

        # A) GROUP (Subfolder) SUMMARIES
        for group, sub_df in df.groupby('Group_Folder'):
            if group == 'Root':
                continue 
            
            # Strict filtering: Must match target_group if provided
            if not create_folder_excels and target_group and group != target_group:
                continue
                
            out_dir = root_dir / group
            safe_group_name = str(Path(group).name).replace('/', '_').replace('\\', '_')
            file_name = f"{safe_group_name}_features_summary.xlsx"
            out_path = out_dir / file_name
            
            os.makedirs(out_dir, exist_ok=True)
            sub_df[export_cols].to_excel(out_path, index=False)
            if target_group:
                print(f"  - Updated specific Group Summary: {out_path}")
            else:
                print(f"  - Group Summary: {out_path}")
            
        # B) INDIVIDUAL VIDEO FOLDER EXCELS
        for _, row in df.iterrows():
            video_name = row['Video_Folder']
            group_name = row['Group_Folder']
            
            # Strict filtering: Must match BOTH video name and group name
            if not create_folder_excels:
                if target_video and video_name != target_video:
                    continue
                if target_group and group_name != target_group:
                    continue
                
            out_dir = Path(row['__Source_Dir'])
            file_name = f"{video_name}_features.xlsx"
            out_path = out_dir / file_name
            
            single_row_df = pd.DataFrame([row[export_cols]])
            single_row_df.to_excel(out_path, index=False)
            if target_video:
                print(f"  - Updated specific Video Excel: {out_path}")

        print(f"Finished updating folder Excel files.")

@click.command()
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--all-excels", is_flag=True, help="Create separate Excel files for EVERY subfolder and individual video folder.")
def cli(results_dir, all_excels):
    """
    Aggregates all extracted feature JSON files into Excel spreadsheets.
    Always creates a Master Excel file in the results directory.
    """
    export_features_to_excel(results_dir, create_folder_excels=all_excels)

if __name__ == '__main__':
    cli()