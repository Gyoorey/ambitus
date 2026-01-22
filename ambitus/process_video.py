import cv2
import numpy as np
import json
import os
import csv

def load_config(config_file="arena_config_poly.json"):
    if not os.path.exists(config_file):
        print(f"HIBA: Nem található a konfigurációs fájl: {config_file}")
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
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(["Time_Sec", "X", "Y"])
            writer.writerows(data)
        print(f"\nSikeres mentés: {filename} ({len(data)} pont)")
    except Exception as e:
        print(f"Hiba a mentéskor: {e}")


def process_video(video_path, show_traj=False, skip_seconds=0.0, min_area=200):
    """
    Process a video file to detect and track rat trajectories within defined polygonal regions.
    This function analyzes video frames to identify moving objects (rats) using background 
    subtraction and morphological operations. Detected positions are recorded with timestamps 
    and saved to a CSV file. Optionally displays real-time tracking visualization.

    Args:
        video_path (str): Path to the video file to be processed.
        show_traj (bool, optional): If True, displays real-time tracking visualization. 
            Defaults to False for faster processing.
        skip_seconds (float, optional): Number of seconds to skip from the beginning of the video. 
            Defaults to 0.0.
        min_area (int, optional): Minimum contour area (in pixels) for object detection. 
            Defaults to 200.
    Returns:
        None
    Notes:
        - Requires a valid configuration file (loaded via load_config()) containing 
          'inner_poly' and 'outer_poly' polygon coordinates.
        - Only detects objects outside the inner polygon boundary.
        - Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) for image enhancement.
        - Uses MOG2 background subtractor for foreground detection.
        - Saves trajectory visualization as 'final_trajectory_map.jpg' if data is detected.
        - Trajectory data includes: (timestamp_seconds, x_coordinate, y_coordinate).
    Raises:
        Returns early if video file cannot be opened or configuration loading fails.
    """
   
    config = load_config()
    if config is None: return

    # Poligon pontok a szűréshez és rajzoláshoz
    inner_pts_array = np.array(config["inner_poly"], dtype=np.int32)
    outer_pts_array = np.array(config["outer_poly"], dtype=np.int32)  # A végső rajzhoz

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Hiba: Videó nem található.")
        return

    # Ugrás a kezdési időpontra
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(skip_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Első releváns képkocka beolvasása (ez lesz a végső háttérkép is)
    ret, first_frame = cap.read()
    if not ret: return

    # Elmentjük az "üres" képet a végére (Deep copy, hogy ne rajzoljunk rá közben)
    final_background_image = first_frame.copy()

    roi_mask = create_poly_mask(config, first_frame.shape)

    # Algoritmusok
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    trajectory_data = []
    frame_counter = start_frame

    print(f"Feldolgozás indítása {skip_seconds} másodperctől...")
    if not show_traj:
        print("Gyors mód aktív (videó megjelenítése kikapcsolva). Kérlek várj...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_counter += 1

        # Egyszerű progress bar gyors módban
        if not show_traj and frame_counter % 100 == 0:
            print(f"\rFeldolgozva: {frame_counter}/{total_frames} frame", end="")

        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_sec = current_time_ms / 1000.0

        # Képfeldolgozás
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        fgmask = fgbg.apply(gray)
        fgmask = cv2.bitwise_and(fgmask, fgmask, mask=roi_mask)

        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy = None, None

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > min_area:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    temp_cx = int(M["m10"] / M["m00"])
                    temp_cy = int(M["m01"] / M["m00"])

                    # Szűrés: Belső poligonon kívül van-e?
                    is_inside_forbidden = cv2.pointPolygonTest(inner_pts_array, (temp_cx, temp_cy), False)
                    if is_inside_forbidden < 0:
                        cx, cy = temp_cx, temp_cy
                        trajectory_data.append((round(timestamp_sec, 3), cx, cy))

        # --- MEGJELENÍTÉS (Kondicionális) ---
        if show_traj:
            # Ha van detektálás, rajzoljunk dobozt
            if cx is not None:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Eddigi pontok kirajzolása
            for point in trajectory_data:
                cv2.circle(frame, (point[1], point[2]), 2, (0, 0, 255), -1)

            # Határok rajzolása
            pts_out = outer_pts_array.reshape((-1, 1, 2))
            pts_in = inner_pts_array.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_out], True, (255, 0, 0), 1)
            cv2.polylines(frame, [pts_in], True, (255, 0, 0), 1)

            cv2.imshow("Tracking (Real-time)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # waitKey(1) a gyorsabb futásért
                print("\nMegszakítva felhasználó által.")
                break

    cap.release()
    if show_traj:
        cv2.destroyAllWindows()

    # --- EREDMÉNYEK MENTÉSE ÉS VÉGSŐ MEGJELENÍTÉS ---

    if len(trajectory_data) > 0:
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        csv_output_file = f"{video_filename}_trajectory.csv"
        save_to_csv(trajectory_data, os.path.join("results", csv_output_file))

        # Végső kép összeállítása az első frame-re
        print("Végső trajektória kirajzolása...")

        # 1. Határok (Kék)
        pts_out = outer_pts_array.reshape((-1, 1, 2))
        pts_in = inner_pts_array.reshape((-1, 1, 2))
        cv2.polylines(final_background_image, [pts_out], True, (255, 0, 0), 2)
        cv2.polylines(final_background_image, [pts_in], True, (255, 0, 0), 2)

        # 2. Útvonal (Piros pontok)
        for point in trajectory_data:
            cv2.circle(final_background_image, (point[1], point[2]), 2, (0, 0, 255), -1)

        # Kép megjelenítése (csak ha show_traj igaz)
        if show_traj:
            cv2.imshow("Vegso Eredmeny (Nyomj gombot a kilepeshez)", final_background_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        # A képet le is mentjük fájlba
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        output_image = os.path.join("results", f"{video_filename}_trajectory_map.jpg")
        cv2.imwrite(output_image, final_background_image)
        print("A végső térkép elmentve: output_image.jpg")
    else:
        print("Nem lett rögzítve adat.")


if __name__ == "__main__":
    process_video("data/1_45.mp4", show_traj=True, skip_seconds=32.0, min_area=200) # Alapértelmezettként mutatja a trajektóriát