import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import deque

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

image = cv.imread('photos/image_1.jpg')

cap = cv.VideoCapture(0)
width, height = 900, 600

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
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = hands.process(frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            bright_factor = is_bright(landmarks, width, height)

            bright_image = image.copy()
            hsv_image = cv.cvtColor(bright_image, cv.COLOR_BGR2HSV)
            h, s, v = cv.split(hsv_image)
            
            # Convert to float to prevent overflow
            v = np.clip(v.astype(np.float32) * bright_factor, 0, 255).astype(np.uint8)

            # Merge back and convert to BGR
            hsv_image = cv.merge((h, s, v))
            bright_image = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

            cv.imshow("Bright Image", bright_image)

    # Convert frame back to BGR for OpenCV display
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    cv.imshow("Camera Feed", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
