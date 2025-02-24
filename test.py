## Still work is there


import mediapipe as mp
import cv2 as cv
import numpy as np
import os
from collections import deque
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)

width = 900
height = 600

position_history = deque(maxlen=5)
last_swipe_time = 0
swipe_threshold = 80
swipe_countdown = 0.8

file_names = os.listdir('photos')
current_image = 0

cv.imshow(str(file_names[current_image]), cv.imread(f"photos/{file_names[current_image]}"))

def swipe_gesture(landmarks):
    index_tip = landmarks.landmark[8]
    index_pip = landmarks.landmark[6]

    middle_tip = landmarks.landmark[12]
    middle_pip = landmarks.landmark[10]

    ring_tip = landmarks.landmark[16]
    pinky_tip = landmarks.landmark[20]

    avg_indx_midd_x = (int(index_tip.x * width) + int(middle_tip.x * width)) // 2

    position_history.append(avg_indx_midd_x)

    indx_middle_distance = np.linalg.norm(np.array([index_tip.x, index_tip.y]) - np.array([middle_tip.x, middle_tip.y]))

    return (index_tip.y < index_pip.y and middle_tip.y < middle_pip.y) and (ring_tip.y > middle_tip.y+0.15 and pinky_tip.y > middle_tip.y+0.18) and (indx_middle_distance < 0.06)


zoom_values = deque(maxlen=5)
def zoom_gesture(landmarks):
    zoom_factor = 0
    global smoothed_distance
    index_tip = landmarks.landmark[8]
    thumb_tip = landmarks.landmark[4]
    thumb_mcp = landmarks.landmark[2]

    middle_tip = landmarks.landmark[12]
    ring_tip = landmarks.landmark[16]
    pinky_tip = landmarks.landmark[20]
    pinky_mcp = landmarks.landmark[17]

    index_w, index_h = int(index_tip.x * width), int(index_tip.y * height)
    thumb_w, thumb_h = int(thumb_tip.x * width), int(thumb_tip.y * height)
    distance = np.linalg.norm(np.array([index_w, index_h]) - np.array([thumb_w, thumb_h]))

    if middle_tip.y > pinky_mcp.y and ring_tip.y > pinky_mcp.y and pinky_tip.y > pinky_mcp.y and thumb_tip.x < thumb_mcp.x:
        new_zoom = np.clip((distance - 20) / 3, 0, 40)

        zoom_values.append(new_zoom)
        zoom_factor = np.mean(zoom_values) 

        zoom_w = int((zoom_factor / 100) * width)
        zoom_h = int((zoom_factor / 100) * height)

        x1, x2 = zoom_w, width - zoom_w
        y1, y2 = zoom_h, height - zoom_h

        if x2 > x1 and y2 > y1:
            return [x1, x2, y1, y2]  
        
bright_values = deque(maxlen=5)

def is_bright(landmarks, width, height):
    """Calculates brightness factor based on hand gestures."""
    bright_factor = 1.0  

    index_tip = landmarks.landmark[8]   
    thumb_tip = landmarks.landmark[4]   
    thumb_mcp = landmarks.landmark[2]   

    middle_tip = landmarks.landmark[12] 
    middle_mcp = landmarks.landmark[10] 
    ring_tip = landmarks.landmark[16]   
    pinky_tip = landmarks.landmark[20]  
    pinky_mcp = landmarks.landmark[17]  

    index_w, index_h = int(index_tip.x * width), int(index_tip.y * height)
    middle_w, middle_h = int(middle_tip.x * width), int(middle_tip.y * height)
    thumb_w, thumb_h = int(thumb_tip.x * width), int(thumb_tip.y * height)

    avg_w, avg_h = (index_w + middle_w) // 2, (index_h + middle_h) // 2

    distance = np.linalg.norm(np.array([avg_w, avg_h]) - np.array([thumb_w, thumb_h]))

    if (ring_tip.y > pinky_mcp.y and pinky_tip.y > pinky_mcp.y and
        thumb_tip.x < thumb_mcp.x and index_tip.y < middle_mcp.y and
        middle_tip.y < middle_mcp.y):
        
        new_brightness = np.clip((distance - 40) / 10, 0.2, 3.0)
        bright_values.append(new_brightness)
        bright_factor = np.mean(bright_values)  
    
    return bright_factor
    
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    
    frame = cv.flip(frame, 1)
    frame = cv.resize(frame, (width, height))
    processed_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(processed_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:           
            if swipe_gesture(landmarks) and len(position_history) == position_history.maxlen:
                diff = position_history[-1] - position_history[0]
                current_time = time.time()

                if current_time - last_swipe_time > swipe_countdown:
                    if diff > swipe_threshold:  
                        if current_image < len(file_names)-1:
                            current_image += 1
                            # print("Right Gesture Detected....", current_image)
                            cv.destroyAllWindows()
                            cv.imshow(str(file_names[current_image]), cv.imread(f"photos/{file_names[current_image]}"))
                            last_swipe_time = current_time
                    elif diff < -swipe_threshold:
                        if current_image > 0:
                            current_image -= 1
                            # print("Left Gesture Detected....", current_image) 
                            cv.destroyAllWindows()
                            cv.imshow(str(file_names[current_image]), cv.imread(f"photos/{file_names[current_image]}"))
                            last_swipe_time = current_time

            zoom = zoom_gesture(landmarks)
            if zoom is not None:
                Cimage = cv.imread(f"photos/{file_names[current_image]}")
                h, w = Cimage.shape[:2]
                img = Cimage[zoom[2]: zoom[3], zoom[0]: zoom[1]]
                img = cv.resize(img, (w, h))
                cv.imshow("Zooming Image", img)


            bright_factor = is_bright(landmarks, width, height)
            if bright_factor:
                bright_image = cv.imread(f"photos/{file_names[current_image]}")
                hsv_image = cv.cvtColor(bright_image, cv.COLOR_BGR2HSV)
                h, s, v = cv.split(hsv_image)
                
                v = np.clip(v.astype(np.float32) * bright_factor, 0, 255).astype(np.uint8)
                hsv_image = cv.merge((h, s, v))
                bright_image = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

                cv.imshow("Image", bright_image)

            cv.putText(frame, "Satisfied....", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    cv.imshow("Frame", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()