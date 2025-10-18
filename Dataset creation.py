import cv2
import numpy as np
import os
from cvzone.HandTrackingModule import HandDetector
import time

# --- CONFIGURATION ---
DATA_FOLDER = "AtoZ"
CLASS_NAME = "A" 
NUM_IMAGES_TO_COLLECT = 200
OFFSET = 29 
IMAGE_SIZE = 400 

save_path = os.path.join(DATA_FOLDER, CLASS_NAME)

# --- SETUP ---
if not os.path.exists(save_path):
    os.makedirs(save_path)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8)

count = len(os.listdir(save_path)) + 1
print(f"Starting collection from: {count}")

# --- SKELETON DRAWING FUNCTION ---
def draw_skeleton_on_white(landmarks, w, h):
    """Creates the 400x400 white background skeleton image."""
    white_img = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8) * 255
    
    # Calculate centering offsets
    os_w = ((IMAGE_SIZE - w) // 2) - 15
    os_h = ((IMAGE_SIZE - h) // 2) - 15

    # Define connections for lines (Simplified list of indices for connections)
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
        (5, 6), (6, 7), (7, 8),         # Index
        (9, 10), (10, 11), (11, 12),    # Middle
        (13, 14), (14, 15), (15, 16),   # Ring
        (17, 18), (18, 19), (19, 20),   # Pinky
        (0, 5), (5, 9), (9, 13), (13, 17), (0, 17) # Palm/Base
    ]
    LINE_COLOR = (0, 255, 0) # Green
    LINE_THICKNESS = 3
    POINT_COLOR = (0, 255, 0) # Green

    for p1_idx, p2_idx in CONNECTIONS:
        pt1 = (landmarks[p1_idx][0] + os_w, landmarks[p1_idx][1] + os_h)
        pt2 = (landmarks[p2_idx][0] + os_w, landmarks[p2_idx][1] + os_h)
        cv2.line(white_img, pt1, pt2, LINE_COLOR, LINE_THICKNESS)

    # Draw circles for landmarks
    for lm in landmarks:
        cv2.circle(white_img, (lm[0] + os_w, lm[1] + os_h), 5, POINT_COLOR, cv2.FILLED)
        
    return white_img

# --- MAIN LOOP ---
while count <= NUM_IMAGES_TO_COLLECT:
    success, img = cap.read()
    if not success: break

    img_display = cv2.flip(img, 1)
    hands, img_display = detector.findHands(img_display, draw=True, flipType=False)
    
    processed_img = None
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        lmList = hand['lmList']
        
        # Crop image area with padding
        y1, y2 = max(0, y - OFFSET), min(img.shape[0], y + h + OFFSET)
        x1, x2 = max(0, x - OFFSET), min(img.shape[1], x + w + OFFSET)
        
        # Recalculate landmarks relative to the cropped area
        relative_lmList = np.array([[lm[0] - x1, lm[1] - y1] for lm in lmList])
        
        # Generate the 400x400 skeleton image
        processed_img = draw_skeleton_on_white(relative_lmList, w, h)
        cv2.imshow("Model Input Preview", processed_img)

    # Display status on live feed
    cv2.putText(img_display, f"Class: {CLASS_NAME}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img_display, f"Count: {count-1} / {NUM_IMAGES_TO_COLLECT}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img_display, "PRESS SPACE TO CAPTURE", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Live Feed", img_display)
    
    # --- Capture Logic ---
    key = cv2.waitKey(1)
    if key == ord(' ') and processed_img is not None:
        file_name = f"{count}.jpg"
        full_file_path = os.path.join(save_path, file_name)
        cv2.imwrite(full_file_path, processed_img)
        print(f"Captured {file_name}")
        count += 1
        time.sleep(0.2) 

    elif key in [ord('q'), 27]: # 'q' or Escape
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()