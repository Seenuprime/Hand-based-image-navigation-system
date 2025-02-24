import cv2 as cv
import mediapipe as mp
import numpy as np
import time
from collections import deque

def detect_swipe(landmarks, width, swipe_threshold=80, swipe_countdown=0.8, position_history_length=5):
    """Detects hand swipes and returns 'right' or 'left'."""

    position_history = deque(maxlen=position_history_length)
    last_swipe_time = 0

    def swipe_gesture(landmarks, width):
        index_tip = landmarks.landmark[8]
        index_pip = landmarks.landmark[6]
        middle_tip = landmarks.landmark[12]
        middle_pip = landmarks.landmark[10]
        ring_tip = landmarks.landmark[16]
        pinky_tip = landmarks.landmark[20]

        avg_indx_midd_x = (int(index_tip.x * width) + int(middle_tip.x * width)) // 2
        position_history.append(avg_indx_midd_x)

        indx_middle_distance = np.linalg.norm(np.array([index_tip.x, index_tip.y]) - np.array([middle_tip.x, middle_tip.y]))

        return (index_tip.y < index_pip.y and middle_tip.y < middle_pip.y) and \
               (ring_tip.y > middle_tip.y + 0.15 and pinky_tip.y > middle_tip.y + 0.18) and \
               (indx_middle_distance < 0.06)

    if swipe_gesture(landmarks, width) and len(position_history) == position_history.maxlen:
        diff = position_history[-1] - position_history[0]
        current_time = time.time()

        if current_time - last_swipe_time > swipe_countdown:
            if diff > swipe_threshold:
                last_swipe_time = current_time
                position_history.clear()
                return "right"
            elif diff < -swipe_threshold:
                last_swipe_time = current_time
                position_history.clear()
                return "left"

    return None

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            width = frame.shape[1]
            swipe_direction = detect_swipe(landmarks, width)

            if swipe_direction == "right":
                print("Right swipe!")
                # Perform right swipe action
            elif swipe_direction == "left":
                print("Left swipe!")
                # Perform left swipe action

    cv.imshow("Hand Tracking", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()