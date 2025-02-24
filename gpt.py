import mediapipe as mp
import cv2 as cv
import numpy as np
import os
from collections import deque
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Camera setup
cap = cv.VideoCapture(0)
width, height = 900, 600

# Gesture tracking
position_history = deque(maxlen=5)
last_swipe_time = 0
swipe_threshold = 80
swipe_cooldown = 0.8
swipe_active = False  # Flag to prevent zooming while swiping

# Image loading
image_folder = 'photos'
file_names = sorted(os.listdir(image_folder))  # Ensure files are in order
current_image = 0

if not file_names:
    print("No images found in the 'photos' directory!")
    cap.release()
    cv.destroyAllWindows()
    exit()

# Zoom parameters
zoom_level = 1.0  # Start with normal zoom
min_zoom, max_zoom = 1.0, 3.0
previous_distance = None  # Track previous pinch distance

# Display the first image
cv.imshow("Image Viewer", cv.imread(f"{image_folder}/{file_names[current_image]}"))


def swipe_gesture(landmarks):
    """Detects a left or right swipe gesture."""
    global swipe_active

    index_tip, index_pip = landmarks.landmark[8], landmarks.landmark[6]
    middle_tip, middle_pip = landmarks.landmark[12], landmarks.landmark[10]
    ring_tip, pinky_tip = landmarks.landmark[16], landmarks.landmark[20]

    avg_index_middle_x = (int(index_tip.x * width) + int(middle_tip.x * width)) // 2
    position_history.append(avg_index_middle_x)

    distance = np.linalg.norm(
        np.array([index_tip.x, index_tip.y]) - np.array([middle_tip.x, middle_tip.y])
    )

    detected = (
        index_tip.y < index_pip.y
        and middle_tip.y < middle_pip.y
        and ring_tip.y > middle_tip.y + 0.15
        and pinky_tip.y > middle_tip.y + 0.18
        and distance < 0.06
    )

    if detected:
        swipe_active = True  # Disable zoom when swipe is detected

    return detected


def zoom_gesture(landmarks):
    """Smooth zooming gesture using pinch distance."""
    global zoom_level, previous_distance, swipe_active

    if swipe_active:  # Prevent zooming while swiping
        return False

    index_tip, thumb_tip = landmarks.landmark[8], landmarks.landmark[4]

    # Calculate pixel positions
    index_x, index_y = int(index_tip.x * width), int(index_tip.y * height)
    thumb_x, thumb_y = int(thumb_tip.x * width), int(thumb_tip.y * height)

    # Compute Euclidean distance between index finger and thumb
    pinch_distance = np.linalg.norm(np.array([index_x, index_y]) - np.array([thumb_x, thumb_y]))

    if previous_distance is None:
        previous_distance = pinch_distance  # Set initial distance
        return False  # Don't zoom yet

    # Compute zoom change
    zoom_change = (pinch_distance - previous_distance) * 0.01  # Smooth scale factor

    # Only update zoom if there's a significant pinch movement
    if abs(zoom_change) > 0.01:
        zoom_level = np.clip(zoom_level + zoom_change, min_zoom, max_zoom)

    # Update previous distance for next iteration
    previous_distance = pinch_distance

    return True  # Zooming detected


def display_image(image_path, zoom):
    """Displays the image with the given zoom level."""
    img = cv.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    h, w = img.shape[:2]
    zoom_w, zoom_h = int(zoom * w), int(zoom * h)

    zoomed_img = cv.resize(img, (zoom_w, zoom_h))

    # Crop to maintain original dimensions
    center_x, center_y = zoom_w // 2, zoom_h // 2
    cropped_img = zoomed_img[
        max(0, center_y - h // 2): min(zoom_h, center_y + h // 2),
        max(0, center_x - w // 2): min(zoom_w, center_x + w // 2),
    ]

    cv.imshow("Image Viewer", cropped_img)


def process_swipe():
    """Processes swipe gestures to navigate through images."""
    global current_image, last_swipe_time, zoom_level, previous_distance, swipe_active

    if len(position_history) < position_history.maxlen:
        return

    diff = position_history[-1] - position_history[0]
    current_time = time.time()

    if current_time - last_swipe_time > swipe_cooldown:
        if diff > swipe_threshold and current_image < len(file_names) - 1:
            current_image += 1  # Swipe right (next image)
        elif diff < -swipe_threshold and current_image > 0:
            current_image -= 1  # Swipe left (previous image)

        # Reset zoom when switching images
        zoom_level = 1.0
        previous_distance = None  # Reset distance tracking
        swipe_active = False  # Allow zooming again

        display_image(f"{image_folder}/{file_names[current_image]}", zoom_level)
        last_swipe_time = current_time


# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(cv.resize(frame, (width, height)), 1)
    processed_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(processed_frame)

    is_zooming = False

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            if swipe_gesture(landmarks):
                process_swipe()

            if not swipe_active:  # Only allow zoom if not swiping
                is_zooming = zoom_gesture(landmarks)

            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    if is_zooming:
        display_image(f"{image_folder}/{file_names[current_image]}", zoom_level)

    cv.putText(frame, "Gesture Control Active", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow("Camera Feed", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
