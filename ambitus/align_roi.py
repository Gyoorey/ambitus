import cv2
import numpy as np
import json
import os
import click

# Az eredeti, jól beállított képen lévő sötét négyzet 4 sarka.
BASE_CORNERS = np.array([
    [323, 138], [511, 148], [506, 336], [314, 327]
], dtype=np.float32)

def transform_polygon(poly_pts, cx0, cy0, cx1, cy1, angle_degrees):
    """ Eltolja és elforgatja a JSON pontokat a talált eltérések alapján. """
    pts = np.array(poly_pts, dtype=np.float32)
    
    # 1. Eltolás az origóba (az eredeti középponthoz képest)
    pts[:, 0] -= cx0
    pts[:, 1] -= cy0
    
    # 2. Forgatás 
    M = cv2.getRotationMatrix2D((0, 0), angle_degrees, 1.0)
    pts_hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    rotated_pts = M.dot(pts_hom.T).T
    
    # 3. Eltolás az új középpontba
    rotated_pts[:, 0] += cx1
    rotated_pts[:, 1] += cy1
    
    return np.int32(np.round(rotated_pts)).tolist()

@click.command()
@click.argument("new_video", type=click.Path(exists=True))
@click.option("--base-image", default="base_reference.jpg", type=click.Path(exists=True), help="Az eredeti referenciakép (A bázis videó első frame-je)")
@click.option("--base-config", default="arena_config_poly.json", help="Az eredeti ROI JSON fájl")
@click.option("--debug", is_flag=True, help="Megjeleníti a vizuális eredményt")
def align(new_video, base_image, base_config, debug):
    if not os.path.exists(base_config):
        print(f"HIBA: A bázis konfiguráció ({base_config}) nem található!")
        return

    # --- 1. KÉP ÉS ÚJ VIDEÓ BEOLVASÁSA ---
    frame_base = cv2.imread(base_image)
    if frame_base is None:
        print(f"HIBA: Nem sikerült beolvasni a referencia képet ({base_image})!")
        return
        
    cap_new = cv2.VideoCapture(new_video)
    ret_n, frame_new = cap_new.read()
    cap_new.release()
    if not ret_n: 
        print("HIBA: Nem sikerült beolvasni az új videó első képkockáját.")
        return

    gray_base = cv2.cvtColor(frame_base, cv2.COLOR_BGR2GRAY)
    gray_new = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_new_clahe = clahe.apply(gray_new)

    # A bázis négyzet befoglaló dobozának kiszámítása
    x, y, w, h = cv2.boundingRect(BASE_CORNERS)
    padding = 15
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(gray_base.shape[1], x + w + padding), min(gray_base.shape[0], y + h + padding)
    
    # Az eredeti középpont (Forgatási és eltolási origó)
    cx0 = x1 + (x2 - x1) / 2.0
    cy0 = y1 + (y2 - y1) / 2.0

    print(f"Keresem a négyzetet az új videón a(z) '{base_image}' alapján...")
    
    best_score = -1
    best_angle = 0
    best_center = (cx0, cy0)
    score_at_zero = 0

    # Végignézzük -10 és +10 fok között
    for angle in range(-10, 11, 1):
        M_rot = cv2.getRotationMatrix2D((cx0, cy0), angle, 1.0)
        # BORDER_REPLICATE: Ha a kép széle befordul, az utolsó pixelt húzza tovább
        rotated_base = cv2.warpAffine(gray_base, M_rot, (gray_base.shape[1], gray_base.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        
        # Kivágjuk a sablont a már elforgatott képből
        template = rotated_base[y1:y2, x1:x2]
        template = clahe.apply(template)
        
        res = cv2.matchTemplate(gray_new_clahe, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        tw = x2 - x1
        th = y2 - y1
        match_cx = max_loc[0] + tw / 2.0
        match_cy = max_loc[1] + th / 2.0

        if angle == 0:
            score_at_zero = max_val
            best_score = max_val
            best_angle = 0
            best_center = (match_cx, match_cy)
        else:
            # "Lustaság" (Penalty) a felesleges forgatások ellen
            angle_penalty = abs(angle) * 0.001
            required_score = score_at_zero + 0.005 + angle_penalty
            
            if max_val > required_score and max_val > best_score:
                best_score = max_val
                best_angle = angle
                best_center = (match_cx, match_cy)

    print(f"Eredmény: {(best_score*100):.1f}%-os egyezés!")
    print(f"Eltolás X: {best_center[0] - cx0:.1f} px, Y: {best_center[1] - cy0:.1f} px. Forgatás: {best_angle} fok.")

    # --- JSON TRANSZFORMÁCIÓ ÉS MENTÉS ---
    with open(base_config, 'r') as f:
        config = json.load(f)

    new_config = {
        "outer_poly": transform_polygon(config["outer_poly"], cx0, cy0, best_center[0], best_center[1], best_angle),
        "inner_poly": transform_polygon(config["inner_poly"], cx0, cy0, best_center[0], best_center[1], best_angle)
    }

    video_filename = os.path.splitext(os.path.basename(new_video))[0]
    output_json = f"{video_filename}_config.json"
    
    with open(output_json, 'w') as f:
        json.dump(new_config, f)

    print(f"Siker! Az eltolt ROI elmentve ide: {output_json}")

    # --- VIZUÁLIS ELLENŐRZÉS ---
    if debug:
        disp = frame_new.copy()
        cv2.drawMarker(disp, (int(best_center[0]), int(best_center[1])), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        pts_out = np.int32(new_config["outer_poly"]).reshape((-1, 1, 2))
        pts_in = np.int32(new_config["inner_poly"]).reshape((-1, 1, 2))
        cv2.polylines(disp, [pts_out], True, (0, 255, 255), 2)
        cv2.polylines(disp, [pts_in], True, (0, 255, 255), 2)

        cv2.imshow("Template Match Alignment", disp)
        print("Nyomj egy gombot a kilépéshez...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    align()