import cv2
import json
import numpy as np
import os

# --- KONFIGURÁCIÓ ---
VIDEO_PATH = "data/45.mp4"
CONFIG_FILE = "arena_config_poly.json"

# Globális változók
selected_points = []  # Ebben tároljuk a (x, y) koordinátákat


def mouse_callback(event, x, y, flags, param):
    global selected_points

    if event == cv2.EVENT_LBUTTONDOWN:
        # Maximum 8 pontot engedünk (4 külső + 4 belső)
        if len(selected_points) < 8:
            selected_points.append((x, y))
            print(f"Pont rögzítve: {x}, {y}")


def main():
    global selected_points

    if not os.path.exists(VIDEO_PATH):
        print(f"HIBA: A '{VIDEO_PATH}' fájl nem található!")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("HIBA: Nem sikerült beolvasni a videó első képkockáját.")
        return

    cv2.namedWindow("Polygon Selector")
    cv2.setMouseCallback("Polygon Selector", mouse_callback)

    print("--- HASZNÁLATI UTASÍTÁS ---")
    print("1. Kattints a KÜLSŐ határ 4 sarkára (körbejárva!).")
    print("2. Kattints a BELSŐ határ 4 sarkára (körbejárva!).")
    print("'r': Újrakezdés (Reset)")
    print("'q' vagy 'Enter': Mentés és kilépés")

    while True:
        img_copy = frame.copy()

        # Pontok kirajzolása
        for pt in selected_points:
            cv2.circle(img_copy, pt, 5, (0, 0, 255), -1)

        # 1. KÜLSŐ Poligon kirajzolása (első 4 pont)
        if len(selected_points) > 0:
            # Vonalak húzása a pontok között, amíg nincs meg a 4.
            limit_outer = min(len(selected_points), 4)
            for i in range(limit_outer - 1):
                cv2.line(img_copy, selected_points[i], selected_points[i + 1], (0, 255, 0), 2)

            # Ha megvan a 4. pont, kössük össze az utolsót az elsővel (zárjuk be)
            if len(selected_points) >= 4:
                cv2.line(img_copy, selected_points[3], selected_points[0], (0, 255, 0), 2)

        # 2. BELSŐ Poligon kirajzolása (második 4 pont)
        if len(selected_points) > 4:
            limit_inner = len(selected_points)
            for i in range(4, limit_inner - 1):
                cv2.line(img_copy, selected_points[i], selected_points[i + 1], (0, 0, 255), 2)

            # Ha megvan a 8. pont, zárjuk be a belső kört is
            if len(selected_points) == 8:
                cv2.line(img_copy, selected_points[7], selected_points[4], (0, 0, 255), 2)

        # Instrukciók
        if len(selected_points) < 4:
            msg = f"KULSO HATAR: {len(selected_points)}/4 pont"
        elif len(selected_points) < 8:
            msg = f"BELSO HATAR: {len(selected_points) - 4}/4 pont"
        else:
            msg = "KESZ! Nyomj 'q'-t a menteshez."

        cv2.putText(img_copy, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Polygon Selector", img_copy)

        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            selected_points = []
            print("Resetelve.")

        if key == ord('q') or key == 13:
            if len(selected_points) == 8:
                data = {
                    "outer_poly": selected_points[:4],
                    "inner_poly": selected_points[4:]
                }
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(data, f)
                print(f"Koordináták elmentve: {CONFIG_FILE}")
                break
            elif key == ord('q'):
                print("Kilépés mentés nélkül.")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()