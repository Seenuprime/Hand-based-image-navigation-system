import mediapipe as mp
import cv2 as cv
import numpy as np
from collections import deque

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Load image
original_image = cv.imread('photos/image_3.webp')
height, width = original_image.shape[:2]

# Start Video Capture
cap = cv.VideoCapture(0)

zoom_values = deque(maxlen=5)  # Store last 5 zoom levels for smoothing

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame = cv.flip(cv.cvtColor(frame, cv.COLOR_BGR2RGB), 1)  # Convert to RGB for Mediapipe
    results = hands.process(frame)

    # Reset zoom image to the original
    image = original_image.copy()
    zoom_factor = 0  # Default zoom

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index and thumb tip coordinates
            index_tip = landmarks.landmark[8]
            thumb_tip = landmarks.landmark[4]

            index_w, index_h = int(index_tip.x * w), int(index_tip.y * h)
            thumb_w, thumb_h = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Calculate Euclidean distance between fingers
            distance = int(np.linalg.norm(np.array([index_w, index_h]) - np.array([thumb_w, thumb_h])))

            # Normalize distance to control zoom percentage (map range 20-150 to 0-50%)
            new_zoom = np.clip((distance - 20) / 3, 0, 50)  # Keep zoom range between 0-50%

            # Add zoom value to buffer for smoothing
            zoom_values.append(new_zoom)
            zoom_factor = np.mean(zoom_values)  # Smooth zoom by averaging last values

            # Calculate zoomed region
            zoom_w = int((zoom_factor / 100) * width)
            zoom_h = int((zoom_factor / 100) * height)

            x1, x2 = zoom_w, width - zoom_w
            y1, y2 = zoom_h, height - zoom_h

            # Ensure valid cropping
            if x2 > x1 and y2 > y1:
                cropped_image = image[y1:y2, x1:x2]
                image = cv.resize(cropped_image, (width, height))

            # Display cropped image
            cv.imshow("Zoomed Image", cv.cvtColor(image, cv.COLOR_RGB2BGR))

    # Convert back to BGR for OpenCV
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    cv.imshow("Hand Tracking", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
