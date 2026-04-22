import os
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
    Rekurzívan kilapítja a JSON dictionary-t.
    ÚJ LOGIKA: Csak a 'Minute_X' és 'Metadata' csoportok nevét tartja meg előtagként,
    mivel ezekben a kulcsok ismétlődnek. A többi főcsoport (I_..., II_...) nevét teljesen eldobja.
    """
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            # Csak a percek és a metaadatok kapnak prefixet
            if k.startswith("Minute_") or k == "Metadata":
                new_parent = f"{k}_"
            else:
                new_parent = "" # Eldobjuk a kategória nevét
                
            items.extend(flatten_dict_smart(v, new_parent).items())
        else:
            final_key = f"{parent_key}{k}"
            items.append((final_key, v))
    return dict(items)

def export_features_to_excel(results_dir, create_folder_excels=False):
    root_dir = Path(results_dir).resolve()
    
    if not root_dir.exists():
        print(f"ERROR: Results directory not found: {root_dir}")
        return

    # Összes feature JSON megkeresése
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
        
        # Tisztított, okos kilapítás
        flat_data = flatten_dict_smart(data)
        
        # Mappa infók kinyerése
        rel_path = jpath.relative_to(root_dir)
        parent_dir = rel_path.parent
        
        video_folder = parent_dir.name
        group_folder = str(parent_dir.parent) if str(parent_dir.parent) != '.' else 'Root'
        
        flat_data['Group_Folder'] = group_folder
        flat_data['Video_Folder'] = video_folder
        
        # Elmentjük az eredeti fájl útvonalát
        flat_data['__Source_Dir'] = str(jpath.parent)
        
        all_data.append(flat_data)

    df = pd.DataFrame(all_data)

    # Oszlopok sorrendjének beállítása
    cols = df.columns.tolist()
    front_cols = ['Group_Folder', 'Video_Folder', 'Metadata_Analyzed_File']
    existing_front_cols = [c for c in front_cols if c in cols]
    
    # Kiszűrjük a segédoszlopokat és a már előre rakott oszlopokat
    rest_cols = [c for c in cols if c not in existing_front_cols and c != '__Source_Dir']
    
    # A végleges megjelenítendő oszloplista
    export_cols = existing_front_cols + rest_cols
    
    # --- 1. MASTER EXCEL LÉTREHOZÁSA ---
    master_out = root_dir / "Ambitus_Master_Features.xlsx"
    df[export_cols].to_excel(master_out, index=False)
    print(f"SUCCESS: Master Excel saved to {master_out}")

    # --- 2. MINDEN MAPPA ÉS ALMAPPA EXCEL LÉTREHOZÁSA (Opcionális) ---
    if create_folder_excels:
        print("\nCreating individual folder Excels...")
        
        # A) ALMAPPA (Csoport) ÖSSZESÍTŐK
        for group, sub_df in df.groupby('Group_Folder'):
            if group == 'Root':
                continue 
                
            out_dir = root_dir / group
            safe_group_name = str(Path(group).name).replace('/', '_').replace('\\', '_')
            file_name = f"{safe_group_name}_features_summary.xlsx"
            out_path = out_dir / file_name
            
            os.makedirs(out_dir, exist_ok=True)
            sub_df[export_cols].to_excel(out_path, index=False)
            print(f"  - Group Summary: {out_path}")
            
        # B) EGYEDI VIDEÓ MAPPA EXCELEK
        for _, row in df.iterrows():
            out_dir = Path(row['__Source_Dir'])
            video_name = row['Video_Folder']
            file_name = f"{video_name}_features.xlsx"
            out_path = out_dir / file_name
            
            single_row_df = pd.DataFrame([row[export_cols]])
            single_row_df.to_excel(out_path, index=False)

        print(f"Finished creating individual folder Excel files.")

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