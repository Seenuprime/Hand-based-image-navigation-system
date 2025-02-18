import cv2 as cv
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)

# Smoothing parameter (adjust for more/less smoothing)
alpha = 0.2
smoothed_distance = None

def simple_zoom(image, landmarks, width, height, zoom_factor=1.2): #Increase zoom factor.
    global smoothed_distance

    if not landmarks:
        return image

    index_tip = landmarks.landmark[8]
    thumb_tip = landmarks.landmark[4]

    index_x, index_y = int(index_tip.x * width), int(index_tip.y * height)
    thumb_x, thumb_y = int(thumb_tip.x * width), int(thumb_tip.y * height)

    distance = np.linalg.norm(np.array([index_x, index_y]) - np.array([thumb_x, thumb_y]))

    # Smoothing with EMA
    if smoothed_distance is None:
        smoothed_distance = distance
    else:
        smoothed_distance = alpha * distance + (1 - alpha) * smoothed_distance

    scaled_distance = smoothed_distance / 30.0  # Adjusted scaling factor.
    zoom_level = 1.0 + (scaled_distance * (zoom_factor - 1.0))
    zoom_level = np.clip(zoom_level, 0.5, 3.0) #increased max zoom.

    new_width = int(image.shape[1] * zoom_level)
    new_height = int(image.shape[0] * zoom_level)

    zoomed_image = cv.resize(image, (new_width, new_height))

    center_x, center_y = new_width // 2, new_height // 2
    cropped_image = zoomed_image[max(0, center_y - image.shape[0] // 2):min(new_height, center_y + image.shape[0] // 2),
                                  max(0, center_x - image.shape[1] // 2):min(new_width, center_x + image.shape[1] // 2)]

    return cropped_image

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    height, width, _ = image.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            zoomed_image = simple_zoom(image, hand_landmarks, width, height)
            cv.imshow("Zoomed Image", zoomed_image)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        cv.imshow("Zoomed Image", image)

    cv.imshow("Original", image)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()