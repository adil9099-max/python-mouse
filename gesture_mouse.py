import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# --- CONFIGURATION ---
CAM_WIDTH, CAM_HEIGHT = 640, 480    # Camera Resolution
FRAME_REDUCTION = 100               # Padding (so you don't have to reach edge of cam)
SMOOTHING = 5                       # Higher = Smoother mouse, but more lag
CLICK_THRESHOLD = 30                # Distance between fingers to trigger click

# --- INITIALIZATION ---
cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Screen Metrics
screen_w, screen_h = pyautogui.size()
plocX, plocY = 0, 0 # Previous Location
clocX, clocY = 0, 0 # Current Location

print("System Active. Use Index finger to point. Pinch Thumb+Index to click.")

while True:
    success, img = cap.read()
    if not success:
        break

    # 1. Find Hand Landmarks
    img = cv2.flip(img, 1) # Mirror the image for natural interaction
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    cv2.rectangle(img, (FRAME_REDUCTION, FRAME_REDUCTION), 
                 (CAM_WIDTH - FRAME_REDUCTION, CAM_HEIGHT - FRAME_REDUCTION),
                 (255, 0, 255), 2)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

            # 2. Get Tip of Index (8) and Thumb (4)
            if len(lm_list) != 0:
                x1, y1 = lm_list[8][1:]  # Index Tip
                x2, y2 = lm_list[4][1:]  # Thumb Tip

                # 3. Check which fingers are up (Basic check)
                # We assume if Index is high, we are in "Move Mode"
                
                # 4. Convert Coordinates (Cam -> Screen)
                # Interpolate from Frame Reduced region to Full Screen
                x3 = np.interp(x1, (FRAME_REDUCTION, CAM_WIDTH - FRAME_REDUCTION), (0, screen_w))
                y3 = np.interp(y1, (FRAME_REDUCTION, CAM_HEIGHT - FRAME_REDUCTION), (0, screen_h))

                # 5. Smoothen Values (Prevents jitter)
                clocX = plocX + (x3 - plocX) / SMOOTHING
                clocY = plocY + (y3 - plocY) / SMOOTHING

                # 6. Move Mouse
                # Use try-catch to prevent crash if mouse hits edge boundary issues
                try:
                    pyautogui.moveTo(clocX, clocY)
                except:
                    pass

                plocX, plocY = clocX, clocY

                # 7. Clicking Mode (Distance between Index and Thumb)
                distance = np.hypot(x2 - x1, y2 - y1)
                
                # Visual Feedback for pinch
                if distance < CLICK_THRESHOLD:
                    cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                    pyautogui.click()
                else:
                    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

            # Draw Skeleton
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    # 8. Display
    cv2.imshow("AI Mouse Controller", img)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
