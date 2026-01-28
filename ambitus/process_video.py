import cv2
import numpy as np
import json
import os
import csv

# --- KONSTANSOK ---
MAX_JUMP_DIST = 100 

class SimpleRatTracker:
    def __init__(self, inner_poly, outer_poly):
        self.name = "Simple (MOG2)"
        self.missed_frames = 0
        self.collision_detected = False
        
        # Ezek hiányoztak, de kellenek a get_best_contour függvénynek:
        self.is_initialized = False
        self.last_valid_pos = None 
        
        self.inner_poly = inner_poly
        self.outer_poly = outer_poly

    def update(self, measured_coords, frame_gray=None):
        if measured_coords is not None:
            self.missed_frames = 0
            
            # Frissítjük az állapotot, hogy a get_best_contour tudja, 
            # hogy már követünk valamit (Distance Gating aktiválása)
            self.is_initialized = True
            self.last_valid_pos = measured_coords
            
            return measured_coords
        else:
            self.missed_frames += 1
            return None

    def get_state_debug(self):
        return 0.0, 0.0, 0.0


class KalmanRatTracker:
    def __init__(self, inner_poly, outer_poly):
        self.name = "Kalman Filter"
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1

        self.last_valid_pos = None
        self.is_initialized = False
        self.missed_frames = 0
        self.collision_detected = False 
        
        self.inner_poly = inner_poly
        self.outer_poly = outer_poly

    def check_geometric_constraints(self, x, y):
        if cv2.pointPolygonTest(self.inner_poly, (float(x), float(y)), False) >= 0: return False 
        if cv2.pointPolygonTest(self.outer_poly, (float(x), float(y)), False) < 0: return False 
        return True

    def update(self, measured_coords, frame_gray=None): # frame_gray a kompatibilitás miatt kell
        pred = self.kalman.predict()
        pred_x, pred_y = int(pred[0, 0]), int(pred[1, 0])
        self.collision_detected = False

        if not self.is_initialized:
            if measured_coords is not None:
                self.kalman.statePre = np.array([[measured_coords[0]], [measured_coords[1]], [0], [0]], np.float32)
                self.kalman.statePost = np.array([[measured_coords[0]], [measured_coords[1]], [0], [0]], np.float32)
                self.last_valid_pos = measured_coords
                self.is_initialized = True
                return measured_coords
            return None

        if measured_coords is not None:
            measurement = np.array([[np.float32(measured_coords[0])], [np.float32(measured_coords[1])]])
            self.kalman.correct(measurement)
            self.last_valid_pos = measured_coords
            self.missed_frames = 0
            return measured_coords
        else:
            self.missed_frames += 1
            if not self.check_geometric_constraints(pred_x, pred_y):
                self.collision_detected = True
                if self.last_valid_pos:
                    self.kalman.statePost[0, 0] = self.last_valid_pos[0]
                    self.kalman.statePost[1, 0] = self.last_valid_pos[1]
                    self.kalman.statePost[2, 0] = 0.0 
                    self.kalman.statePost[3, 0] = 0.0 
                    return self.last_valid_pos
            
            decay_factor = 0.9 
            self.kalman.statePost[2, 0] *= decay_factor
            self.kalman.statePost[3, 0] *= decay_factor

            if self.missed_frames > 50: return None
            return (pred_x, pred_y)

    def get_state_debug(self):
        if not self.is_initialized: return 0.0, 0.0, 0.0
        vx, vy = self.kalman.statePost[2, 0], self.kalman.statePost[3, 0]
        return np.sqrt(vx**2 + vy**2), vx, vy


class OpticalFlowRatTracker:
    def __init__(self, inner_poly, outer_poly):
        self.name = "Optical Flow"
        self.last_valid_pos = None
        self.is_initialized = False
        self.missed_frames = 0
        self.collision_detected = False
        
        self.inner_poly = inner_poly
        self.outer_poly = outer_poly
        
        # Optical Flow változók
        self.prev_gray = None
        self.p0 = None # Követendő pontok
        
        # Lucas-Kanade paraméterek
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Feature keresés paraméterek (GoodFeaturesToTrack)
        self.feature_params = dict(maxCorners=20, qualityLevel=0.3, minDistance=7, blockSize=7)

    def check_geometric_constraints(self, x, y):
        if cv2.pointPolygonTest(self.inner_poly, (float(x), float(y)), False) >= 0: return False 
        if cv2.pointPolygonTest(self.outer_poly, (float(x), float(y)), False) < 0: return False 
        return True

    def update(self, measured_coords, frame_gray):
        """
        Ha van measured_coords (MOG2): Újrainicializáljuk a pontokat a patkányon.
        Ha nincs: Optical Flow-val követjük a régi pontokat.
        """
        self.collision_detected = False
        current_pos = None

        # 1. ESET: Van erős detektálás (MOG2) -> Ez a "Ground Truth"
        if measured_coords is not None:
            self.last_valid_pos = measured_coords
            self.missed_frames = 0
            self.is_initialized = True
            current_pos = measured_coords
            
            # Keressünk új követési pontokat a patkány körül, hogy felkészüljünk a következő (esetleg hiányos) frame-re
            # Csinálunk egy kis maszkot a patkány köré
            mask_roi = np.zeros_like(frame_gray)
            cx, cy = measured_coords
            # Kb 40x40-es doboz a patkány körül feature kereséshez
            cv2.circle(mask_roi, (cx, cy), 20, 255, -1) 
            
            self.p0 = cv2.goodFeaturesToTrack(frame_gray, mask=mask_roi, **self.feature_params)
            self.prev_gray = frame_gray.copy()
            
            return current_pos

        # 2. ESET: Nincs detektálás -> Optical Flow
        if self.is_initialized and self.p0 is not None and len(self.p0) > 0:
            self.missed_frames += 1
            
            # Kiszámoljuk az elmozdulást
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, self.p0, None, **self.lk_params)
            
            # Kiválasztjuk a jó pontokat (amiket sikerült követni)
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = self.p0[st == 1]
                
                if len(good_new) > 0:
                    # Kiszámoljuk az átlagos elmozdulást (dx, dy)
                    displacement = good_new - good_old
                    dx = np.median(displacement[:, 0])
                    dy = np.median(displacement[:, 1])
                    
                    # Ha nagyon kevés a pont, vagy ugrál, feltételezzük a 0 mozgást (stop)
                    if abs(dx) < 0.5: dx = 0
                    if abs(dy) < 0.5: dy = 0
                    
                    # Frissítjük a pozíciót
                    old_x, old_y = self.last_valid_pos
                    new_x = int(old_x + dx)
                    new_y = int(old_y + dy)
                    
                    # Geometriai ellenőrzés
                    if self.check_geometric_constraints(new_x, new_y):
                        self.last_valid_pos = (new_x, new_y)
                        current_pos = (new_x, new_y)
                        
                        # Frissítjük a pontokat a köv. körre
                        self.p0 = good_new.reshape(-1, 1, 2)
                        self.prev_gray = frame_gray.copy()
                    else:
                        self.collision_detected = True
                        # Falnak mentünk -> Megtartjuk a régit
                        current_pos = self.last_valid_pos
                else:
                    # Elvesztettük a pontokat
                    current_pos = self.last_valid_pos # Jobb híján maradunk helyben
            else:
                current_pos = self.last_valid_pos

            if self.missed_frames > 50: return None
            return current_pos

        return None # Még nincs inicializálva

    def get_state_debug(self):
        # Az OF nem tárol sebességet közvetlenül, de visszaadhatunk 0-t vagy dummy adatot
        return 0.0, 0.0, 0.0


def load_config(config_file="arena_config_poly.json"):
    if not os.path.exists(config_file):
        return None
    with open(config_file, 'r') as f:
        return json.load(f)

def create_poly_mask(config, frame_shape):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    outer_pts = np.array(config["outer_poly"], dtype=np.int32)
    inner_pts = np.array(config["inner_poly"], dtype=np.int32)
    cv2.fillPoly(mask, [outer_pts], 255)
    cv2.fillPoly(mask, [inner_pts], 0)
    return mask

def save_to_csv(data, filename):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(["Time_Sec", "X", "Y"])
            writer.writerows(data)
        print(f"\nSikeres mentés: {filename} ({len(data)} pont)")
    except Exception as e:
        print(f"Hiba a mentéskor: {e}")

def get_best_contour(contours, tracker, inner_pts_array, min_area):
    valid_candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area: continue
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if cv2.pointPolygonTest(inner_pts_array, (cx, cy), False) >= 0: continue
        valid_candidates.append((c, (cx, cy), area))

    if not valid_candidates: return None, None

    if not tracker.is_initialized or tracker.last_valid_pos is None:
        best = max(valid_candidates, key=lambda x: x[2])
        return best[0], best[1]

    last_x, last_y = tracker.last_valid_pos
    best_candidate = None
    min_dist = float('inf')

    for c, (cx, cy), area in valid_candidates:
        dist = np.sqrt((cx - last_x) ** 2 + (cy - last_y) ** 2)
        if dist < min_dist:
            min_dist = dist
            best_candidate = (c, (cx, cy))

    if min_dist > MAX_JUMP_DIST: return None, None
    return best_candidate[0], best_candidate[1]

def clean_trajectory(data, max_drift_dist=50):
    """
    Kiszűri a Kalman-filter "túllövéseit" (overshoot).
    data formátum: [(time, x, y, is_predicted), ...]
    
    Logika: Ha van egy predikált szakasz (csupa True), megnézzük, honnan indult
    és hova érkezett vissza a valós mérés. Ha a patkány valójában nem mozdult el
    (start és end közel van), de a predikció elszállt, akkor töröljük a predikciót.
    """
    if not data: return []
    
    cleaned_data = []
    n = len(data)
    i = 0
    
    while i < n:
        # Ha ez egy mért pont (nem predikció), megtartjuk
        if not data[i][3]: # is_predicted == False
            cleaned_data.append(data[i])
            i += 1
            continue
            
        # Ha predikciót találunk, keressük meg a szakasz végét
        j = i
        while j < n and data[j][3]: # Amíg predikció
            j += 1
            
        # Most az i-től (j-1)-ig tart a predikált szakasz.
        # A j. elem (ha létezik) az első újabb VALÓS mérés.
        # Az (i-1). elem az utolsó VALÓS mérés.
        
        start_valid = data[i-1] if i > 0 else None
        end_valid = data[j] if j < n else None
        
        keep_segment = True
        
        # Csekkoljuk a "driftet"
        if start_valid and end_valid:
            print("Start Valid:", start_valid, "End Valid:", end_valid)
            sx, sy = start_valid[1], start_valid[2]
            ex, ey = end_valid[1], end_valid[2]
            
            # Tényleges elmozdulás a valóságban
            real_displacement = np.sqrt((ex-sx)**2 + (ey-sy)**2)
            
            # Ha a patkány alig mozdult a valóságban, de a tracker sokat jósolt,
            # akkor ez csak egy "megállás közbeni sodródás". Eldobjuk.
            if real_displacement < max_drift_dist:
                keep_segment = False
        
        # Ha nincs vége a szakasznak (videó vége), vagy nincs eleje,
        # akkor dönthetünk úgy, hogy eldobjuk a bizonytalan véget.
        elif start_valid and not end_valid:
             print("No end valid point; discarding trailing drift segment.")
             keep_segment = False # A videó végén kiszaladt a falnak -> kuka
             
        if keep_segment:
            print("Keeping drift segment from", i, "to", j-1)
            # Megtartjuk a pontokat (esetleg simíthatnánk is, de most csak másoljuk)
            for k in range(i, j):
                cleaned_data.append(data[k])
        else:
            print("Discarding drift segment from", i, "to", j-1)
            # Nem tartjuk meg (töröljük a driftet). 
            # Opcionális: interpolálhatnánk a helyét (start és end közé),
            # de a "megállás" esetén a törlés a helyes, mert akkor nem rajzol vonalat.
            pass
            
        i = j # Folytatás a szakasz után

    return cleaned_data

# --- DEBUG NÉZET ÉPÍTŐ ---
def create_debug_view(frame, fgmask, tracker, measured_coords, final_pos):
    """
    Összeállítja a kombinált képet:
    [ Eredeti Videó ] [ Maszk ]
                      [ Info Panel ]
    """
    h, w = frame.shape[:2]
    panel_width = 400 
    
    mask_bgr = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    target_h = int(h / 2)
    mask_resized = cv2.resize(mask_bgr, (panel_width, target_h))
    cv2.rectangle(mask_resized, (0,0), (panel_width-1, target_h-1), (100,100,100), 2)
    cv2.putText(mask_resized, "MOG2 Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    info_panel = np.zeros((h - target_h, panel_width, 3), dtype=np.uint8)
    
    speed, vx, vy = tracker.get_state_debug()
    missed = tracker.missed_frames
    
    line_h = 30
    y_off = 30
    
    status_color = (0, 255, 0)
    status_text = "TRACKING"
    
    if missed > 0:
        status_color = (0, 255, 255)
        status_text = "COASTING"
    if missed > 40:
        status_color = (0, 0, 255)
        status_text = "LOST"
    # Simple tracker esetén nincs predikció, tehát ha missed > 0, az simán "NO DETECTION"
    if tracker.name == "Simple (MOG2)" and missed > 0:
        status_text = "NO DETECTION"
        status_color = (0, 0, 255)
        
    if tracker.collision_detected:
        status_text = "WALL COLLISION"
        status_color = (0, 0, 255)

    cv2.putText(info_panel, f"METHOD: {tracker.name}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) # ÚJ: Módszer kiírása
    y_off += line_h
    cv2.putText(info_panel, f"STATUS: {status_text}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    y_off += line_h
    
    meas_str = f"Meas: {measured_coords}" if measured_coords else "Meas: None"
    cv2.putText(info_panel, meas_str, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 1)
    y_off += line_h
    
    # ITT A VÁLTOZÁS: tracker.name használata
    track_str = f"{tracker.name}: {final_pos}" if final_pos else f"{tracker.name}: None"
    cv2.putText(info_panel, track_str, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
    y_off += line_h * 2
    
    cv2.putText(info_panel, f"Speed: {speed:.2f} px/fr", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
    y_off += line_h
    # ... (maradék változatlan) ...
    
    right_sidebar = np.vstack([mask_resized, info_panel])
    if right_sidebar.shape[0] != h:
        right_sidebar = cv2.resize(right_sidebar, (panel_width, h))
        
    combined = np.hstack([frame, right_sidebar])
    return combined

def process_video(video_path, method='kf', show_traj=False, debug=False, skip_seconds=0.0, min_area=200):
    """
    method: 'kf' (Kalman Filter) vagy 'of' (Optical Flow)
    """
    config = load_config()
    if config is None: return

    inner_pts_array = np.array(config["inner_poly"], dtype=np.int32)
    outer_pts_array = np.array(config["outer_poly"], dtype=np.int32)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(skip_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, first_frame = cap.read()
    if not ret: return

    final_background_image = first_frame.copy()
    roi_mask = create_poly_mask(config, first_frame.shape)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    fgbg = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=30, detectShadows=False)
    
    # --- TRACKER KIVÁLASZTÁSA ---
    if method == 'of':
        print("Using Optical Flow Tracker")
        tracker = OpticalFlowRatTracker(inner_poly=inner_pts_array, outer_poly=outer_pts_array)
    elif method == 'simple':
        print("Using Simple Tracker (MOG2 Only)")
        tracker = SimpleRatTracker(inner_poly=inner_pts_array, outer_poly=outer_pts_array)
    elif method == 'kf':
        print("Using Kalman Filter Tracker")
        tracker = KalmanRatTracker(inner_poly=inner_pts_array, outer_poly=outer_pts_array)
    # ----------------------------

    trajectory_data = []
    frame_counter = start_frame
    visualization_active = show_traj or debug

    print(f"Feldolgozás indítása {skip_seconds} másodperctől...")
    if not visualization_active:
        print("Gyors mód aktív...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_counter += 1
        if not visualization_active and frame_counter % 100 == 0:
            print(f"\rFeldolgozva: {frame_counter}/{total_frames} frame", end="")

        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_sec = current_time_ms / 1000.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        gray_blurred = cv2.GaussianBlur(gray, (7, 7), 0) # MOG2-nek

        fgmask = fgbg.apply(gray_blurred, learningRate=0.0005)
        fgmask = cv2.bitwise_and(fgmask, fgmask, mask=roi_mask)

        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour, measured_coords = get_best_contour(contours, tracker, inner_pts_array, min_area)
        
        # FONTOS: Az update most már kéri a gray frame-et is (az OF miatt)
        # A Kalman tracker szimplán figyelmen kívül hagyja
        final_pos = tracker.update(measured_coords, frame_gray=gray)

        final_cx, final_cy = None, None
        if final_pos is not None:
            final_cx, final_cy = int(final_pos[0]), int(final_pos[1])
            is_predicted = (measured_coords is None)
            
            # Mentés
            if cv2.pointPolygonTest(inner_pts_array, (final_cx, final_cy), False) < 0:
                trajectory_data.append((round(timestamp_sec, 3), final_cx, final_cy, is_predicted))
            else:
                final_cx, final_cy = None, None

        # --- MEGJELENÍTÉS ---
        if visualization_active:
            # 1. Nyers detektálás doboza (Zöld) - Csak ha van kontúr
            if best_contour is not None:
                x, y, w, h = cv2.boundingRect(best_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 2. Tracker pozíciója (Kereszt)
            if final_cx is not None:
                # SZÍNEZÉS: Ha measured_coords None, akkor ez PREDIKCIÓ -> SÁRGA
                # Ha van measured_coords, akkor MÉRÉS -> PIROS
                is_prediction = (measured_coords is None)
                color = (0, 255, 255) if is_prediction else (0, 0, 255) # Sárga vs Piros
                
                cv2.drawMarker(frame, (final_cx, final_cy), color, cv2.MARKER_CROSS, 20, 2)
                
                # Opcionális: kiírhatjuk mellé, hogy "Pred"
                if is_prediction and debug:
                    cv2.putText(frame, "Pred", (final_cx+10, final_cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 3. Trajektória "csík" (Múltbeli pontok)
            # Csak az utolsó 100 pontot rajzoljuk ki a gyorsaság miatt
            for point in trajectory_data[-100:]:
                # point struktúra: (time, x, y, is_predicted)
                p_x, p_y = point[1], point[2]
                p_is_pred = point[3]
                
                pt_color = (0, 255, 255) if p_is_pred else (0, 0, 255)
                cv2.circle(frame, (p_x, p_y), 2, pt_color, -1)

            # Poligonok rajzolása (Kékessárga)
            pts_out = outer_pts_array.reshape((-1, 1, 2))
            pts_in = inner_pts_array.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_out], True, (255, 255, 0), 1)
            cv2.polylines(frame, [pts_in], True, (255, 255, 0), 1)

            if debug:
                display_frame = create_debug_view(frame, fgmask, tracker, measured_coords, (final_cx, final_cy))
            else:
                display_frame = frame

            cv2.imshow("Tracking", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nMegszakítva.")
                break

    cap.release()
    if visualization_active: cv2.destroyAllWindows()

    if len(trajectory_data) > 0:
        # trajectory_data = clean_trajectory(trajectory_data, max_drift_dist=100)
        final_csv_data = [(t, x, y) for (t, x, y, is_pred) in trajectory_data]

        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        # Filenévbe tesszük a módszert (kf vagy of)
        csv_output_file = f"{video_filename}_{method}_trajectory.csv"
        output_path = os.path.join("results", csv_output_file)
        
        save_to_csv(final_csv_data, output_path)

        # Térkép rajzolás...
        print("Végső trajektória kirajzolása...")
        pts_out = outer_pts_array.reshape((-1, 1, 2))
        pts_in = inner_pts_array.reshape((-1, 1, 2))
        cv2.polylines(final_background_image, [pts_out], True, (255, 0, 0), 2)
        cv2.polylines(final_background_image, [pts_in], True, (255, 0, 0), 2)

        # SZÍNEZETT PONTOK RAJZOLÁSA A TÉRKÉPRE
        for point in trajectory_data: # Itt a tisztított adatokat vagy a nyerset használod, attól függően hol tart a kód
            p_x, p_y = point[1], point[2]
            
            # Ellenőrizzük, hogy van-e 4. elem (is_predicted)
            if len(point) > 3:
                p_is_pred = point[3]
                color = (0, 255, 255) if p_is_pred else (0, 0, 255) # Sárga vs Piros
            else:
                color = (0, 0, 255) # Ha nincs infó, legyen piros
                
            cv2.circle(final_background_image, (p_x, p_y), 2, color, -1)

        output_image = os.path.join("results", f"{video_filename}_{method}_trajectory_map.jpg")
        cv2.imwrite(output_image, final_background_image)
        print(f"Kész. ({method})")
    else:
        print("Nem lett rögzítve adat.")