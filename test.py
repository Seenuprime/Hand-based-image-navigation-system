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

def zoom_gesture(landmarks):
    pass

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

            elif zoom_gesture(landmarks):
                pass
            
            cv.putText(frame, "Satisfied....", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    cv.imshow("Frame", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()