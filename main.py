import numpy as np
import cv2
import mediapipe as mp
from subprocess import call

# Create useful objects from mediapipe
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles
model = mpHands.Hands()

# Fire up OpenCV (try changing the argument to 0 or 1 if not working)
cap = cv2.VideoCapture(0)

# Loop
with mpHands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as model:
    while cap.isOpened():
        # Read from camera
        success, image = cap.read()

        # Ignore bad frames, happens sometimes, not frequently
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Get height and widht from image, and convert it to RGB for Hands() model to use it
        h, w, _ = image.shape
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.process(imageRGB)

        # Draw the hand annotations on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb_tip = hand_landmarks.landmark[4]  # accordint to the documentation
                thumb_tip_x, thumb_tip_y = int(thumb_tip.x*w), int(thumb_tip.y*h)

                index_finger_tip = hand_landmarks.landmark[8]   # accordint to the documentation
                index_finger_tip_x, index_finger_tip_y = int(index_finger_tip.x*w), int(index_finger_tip.y*h)

                # Since distance from thumb to index differs depending on how close you are to the camera
                # range [20, 200] is determined ad hoc and should work just fine
                length = np.linalg.norm((index_finger_tip_x-thumb_tip_x, index_finger_tip_y-thumb_tip_y))
                volume = np.interp(length, [20, 200], [0, 100])
                
                # Control volume
                call(["amixer", "-D", "pulse", "sset", "Master", f"{volume}%"])

                # Draw annotations on the image
                cv2.circle(image, (thumb_tip_x, thumb_tip_y), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(image, (index_finger_tip_x, index_finger_tip_y), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(image, (index_finger_tip_x, index_finger_tip_y), (thumb_tip_x, thumb_tip_y), (255, 0, 255), 3)
                mpDraw.draw_landmarks(image, hand_landmarks, mpHands.HAND_CONNECTIONS)

        # Flip the image horizontally for a selfie-view display
        cv2.imshow('Camera feed', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27: # wait for ESC
            break

cap.release()