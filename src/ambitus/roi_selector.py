import cv2
import json
import numpy as np
import os

# Globális változók (modul szinten, mert a callback csak így éri el)
points_current = []
points_outer = []
points_inner = []
current_mode = "OUTER" # OUTER -> INNER -> DONE

def mouse_callback(event, x, y, flags, param):
    global points_current
    
    if current_mode == "DONE": return

    if event == cv2.EVENT_LBUTTONDOWN:
        points_current.append((x, y))
        print(f"Pont: {x}, {y}")

def select_roi(video_path, config_file="arena_config_poly.json"):
    global current_mode, points_outer, points_inner, points_current
    
    # Reset
    points_current = []
    points_outer = []
    points_inner = []
    current_mode = "OUTER"

    if not os.path.exists(video_path):
        print(f"HIBA: A '{video_path}' fájl nem található!")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("HIBA: Nem sikerült megnyitni a videót.")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("HIBA: Nem sikerült beolvasni az első képkockát.")
        return

    window_name = f"ROI Selector - {os.path.basename(video_path)}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"--- ROI KIVÁLASZTÁS: {video_path} ---")
    print("1. KÜLSŐ határ (Tetszőleges számú pont). ENTER a befejezéshez.")
    print("2. BELSŐ határ (Tetszőleges számú pont). ENTER a mentéshez.")
    print("'r': Reset (Újrakezdés)")
    print("'q': Kilépés mentés nélkül")

    while True:
        img_copy = frame.copy()

        # Rajzolás
        poly_list = []
        if current_mode == "OUTER":
            poly_list = [points_outer, points_current] # Már kész (üres) + Épp rajzolt
        elif current_mode == "INNER":
            poly_list = [points_outer, points_current] # Már kész (külső) + Épp rajzolt (belső)
        else:
            poly_list = [points_outer, points_inner]

        # Kirajzoljuk a kész poligonokat és az épp rajzoltat is
        if len(points_outer) > 0:
            pts = np.array(points_outer, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_copy, [pts], True, (0, 255, 0), 2)
            
        if len(points_inner) > 0:
            pts = np.array(points_inner, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_copy, [pts], True, (0, 0, 255), 2)
            
        # Az aktuális (még nyitott) poligon rajzolása
        if len(points_current) > 0:
            for pt in points_current:
                cv2.circle(img_copy, pt, 4, (255, 255, 0), -1)
            if len(points_current) > 1:
                pts = np.array(points_current, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_copy, [pts], False, (255, 255, 0), 1)

        # Üzenetek
        if current_mode == "OUTER":
            msg = f"1. KULSO HATAR ({len(points_current)} pont). ENTER->Next"
            color = (0, 255, 0)
        elif current_mode == "INNER":
            msg = f"2. BELSO HATAR ({len(points_current)} pont). ENTER->Save"
            color = (0, 0, 255)
        else:
            msg = "KESZ! Mentes..."
            color = (0, 255, 255)

        cv2.putText(img_copy, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow(window_name, img_copy)

        key = cv2.waitKey(20) & 0xFF

        # Reset ('r')
        if key == ord('r'):
            points_current = []
            points_outer = []
            points_inner = []
            current_mode = "OUTER"
            print("Resetelve.")

        # Next / Save (ENTER)
        if key == 13: 
            if current_mode == "OUTER":
                if len(points_current) < 3:
                    print("Hiba: Legalább 3 pont kell!")
                else:
                    points_outer = list(points_current) # Mentés
                    points_current = [] # Törlés a következőhöz
                    current_mode = "INNER"
                    print("--> Külső kész. Jöhet a BELSŐ.")
            
            elif current_mode == "INNER":
                if len(points_current) < 3:
                    print("Hiba: Legalább 3 pont kell!")
                else:
                    points_inner = list(points_current)
                    current_mode = "DONE"
                    
                    # JSON Mentés
                    data = {
                        "outer_poly": points_outer,
                        "inner_poly": points_inner
                    }
                    with open(config_file, 'w') as f:
                        json.dump(data, f)
                    print(f"Konfiguráció elmentve: {config_file}")
                    break

        # Quit ('q')
        if key == ord('q'):
            print("Kilépés mentés nélkül.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Teszteléshez (ha közvetlenül hívod)
    select_roi("data/45.mp4")