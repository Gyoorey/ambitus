import cv2
import numpy as np
import json
import os
import imageio

# The 4 corners of the dark square on the original, well-calibrated image.
# We still use these as the mathematical center/origin for our rotations!
BASE_CORNERS = np.array([
    [323, 138], [511, 148], [506, 336], [314, 327]
], dtype=np.float32)

def transform_polygon(poly_pts, cx0, cy0, cx1, cy1, angle_degrees):
    """ Translates and rotates JSON points based on found deviations. """
    pts = np.array(poly_pts, dtype=np.float32)
    
    # 1. Translation to origin (relative to original center)
    pts[:, 0] -= cx0
    pts[:, 1] -= cy0
    
    # 2. Rotation
    M = cv2.getRotationMatrix2D((0, 0), angle_degrees, 1.0)
    pts_hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    rotated_pts = M.dot(pts_hom.T).T
    
    # 3. Translation to new center
    rotated_pts[:, 0] += cx1
    rotated_pts[:, 1] += cy1
    
    return np.int32(np.round(rotated_pts)).tolist()

def align_video_roi(new_video, base_image="base_reference.jpg", base_config="arena_config_poly.json", debug=False, output_dir=None):
    """
    Template matching-based automatic ROI alignment using a LARGE MACRO-TEMPLATE.
    It captures the central square AND the 8 feeding windows for extreme accuracy.
    """
    if not os.path.exists(base_config):
        print(f"ERROR: Base configuration ({base_config}) not found!")
        return None

    frame_base = cv2.imread(base_image)
    if frame_base is None:
        print(f"ERROR: Failed to read reference image ({base_image})!")
        return None
        
    # Read the first frame of the new video
    if new_video.lower().endswith('.mpg'):
        try:
            # PERFECT FFmpeg DE-INTERLACING & SCALING (from your code)
            input_params = []
            output_params = [
                '-vf', 'yadif,scale=768:576,setsar=1',
            ]
            reader = imageio.get_reader(new_video, 'ffmpeg', 
                                        input_params=input_params, 
                                        output_params=output_params)
            frame_new = reader.get_data(0)
            frame_new = cv2.cvtColor(frame_new, cv2.COLOR_RGB2BGR)
            reader.close()
            ret_n = True
        except:
            ret_n = False
    else:
        cap_new = cv2.VideoCapture(new_video)
        ret_n, frame_new = cap_new.read()
        cap_new.release()

    if not ret_n: 
        print("ERROR: Failed to read the first frame of the new video.")
        return None

    gray_base = cv2.cvtColor(frame_base, cv2.COLOR_BGR2GRAY)
    gray_new = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)

    # We apply a slight blur. This makes the matching ignore tiny details 
    # (like floor scratches or a rat) and focus on the large shapes (feeding windows).
    gray_base = cv2.GaussianBlur(gray_base, (5, 5), 0)
    gray_new = cv2.GaussianBlur(gray_new, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_new_clahe = clahe.apply(gray_new)

    # --- MACRO-TEMPLATE SIZING ---
    x, y, w, h = cv2.boundingRect(BASE_CORNERS)
    
    # HUGE padding: 140 pixels. This will include all 8 feeding windows around the center!
    padding = 140 
    
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(gray_base.shape[1], x + w + padding), min(gray_base.shape[0], y + h + padding)
    
    # The mathematical center of our template
    cx0 = x1 + (x2 - x1) / 2.0
    cy0 = y1 + (y2 - y1) / 2.0

    print(f"Searching for the macro-pattern (center + feeding windows) based on '{os.path.basename(base_image)}'...")
    
    best_score = -1
    best_angle = 0
    best_center = (cx0, cy0)
    score_at_zero = 0

    for angle in range(-10, 11, 1):
        # Rotate the entire base image
        M_rot = cv2.getRotationMatrix2D((cx0, cy0), angle, 1.0)
        rotated_base = cv2.warpAffine(gray_base, M_rot, (gray_base.shape[1], gray_base.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        
        # Crop the large template AFTER rotation
        template = rotated_base[y1:y2, x1:x2]
        template = clahe.apply(template)
        
        # Cross-correlation matching
        res = cv2.matchTemplate(gray_new_clahe, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
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
            # Angle penalty to prevent unnecessary 1-degree rotations due to pixelation
            angle_penalty = abs(angle) * 0.001
            required_score = score_at_zero + 0.005 + angle_penalty
            
            if max_val > required_score and max_val > best_score:
                best_score = max_val
                best_angle = angle
                best_center = (match_cx, match_cy)

    print(f"Result: {(best_score*100):.1f}% structural match!")
    print(f"Offset X: {best_center[0] - cx0:.1f} px, Y: {best_center[1] - cy0:.1f} px. Rotation: {best_angle} degrees.")

    with open(base_config, 'r') as f:
        config = json.load(f)

    new_config = {
        "outer_poly": transform_polygon(config["outer_poly"], cx0, cy0, best_center[0], best_center[1], best_angle),
        "inner_poly": transform_polygon(config["inner_poly"], cx0, cy0, best_center[0], best_center[1], best_angle)
    }

    video_filename = os.path.splitext(os.path.basename(new_video))[0]
    
    if output_dir is None:
        output_dir = os.path.join("results", video_filename)
        
    os.makedirs(output_dir, exist_ok=True)
    output_json = os.path.join(output_dir, f"{video_filename}_config.json")
    
    with open(output_json, 'w') as f:
        json.dump(new_config, f)

    print(f"Success! Aligned ROI saved to: {output_json}")

    if debug:
        disp = frame_new.copy()
        
        # --- DRAWINGS ---
        # 1. The new center point (red cross)
        cv2.drawMarker(disp, (int(best_center[0]), int(best_center[1])), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        
        # 2. THE MACRO-TEMPLATE BOUNDARIES (Translated and ROTATED!)
        template_corners = [
            [x1, y1],             # Top-Left
            [x2, y1],             # Top-Right
            [x2, y2],             # Bottom-Right
            [x1, y2]              # Bottom-Left
        ]
        
        # We pass them to our own transform_polygon function, which translates and rotates based on best_angle!
        warped_template_corners = transform_polygon(template_corners, cx0, cy0, best_center[0], best_center[1], best_angle)
        
        # Convert the returned [4x2] list to OpenCV polygon format
        pts_macro = np.int32(warped_template_corners).reshape((-1, 1, 2))
        cv2.polylines(disp, [pts_macro], True, (0, 255, 0), 2)
        
        # Write the label above the top-left corner of the rotated box
        top_left_pt = tuple(warped_template_corners[0])
        cv2.putText(disp, f"Matched Macro-Pattern (Rot: {best_angle} deg)", (top_left_pt[0], top_left_pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 3. The JSON polygons (in yellow)
        pts_out = np.int32(new_config["outer_poly"]).reshape((-1, 1, 2))
        pts_in = np.int32(new_config["inner_poly"]).reshape((-1, 1, 2))
        cv2.polylines(disp, [pts_out], True, (0, 255, 255), 2)
        cv2.polylines(disp, [pts_in], True, (0, 255, 255), 2)

        cv2.imshow("Macro-Template Match Alignment", disp)
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return output_json