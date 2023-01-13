import cv2
import mediapipe as mp
import pandas as pd
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
import numpy as np

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
index = 0

# df = pd.read_csv('data.csv')
df = pd.DataFrame
# df.columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
data = []


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.

    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

    mp_drawing.draw_landmarks(
      image,
      results.right_hand_landmarks,
      mp_holistic.HAND_CONNECTIONS
    )

   
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)
    image_height, image_width, _ = image.shape
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        # Here is How to Get All the Coordinates
        for ids, landmrk in enumerate(hand_landmarks.landmark):
            # print(ids)
            # print(landmrk)
            # data.append(landmrk)
            cx, cy = landmrk.x * image_width, landmrk.y*image_height

            wristZ = landmrk.z * image_width
            cz = wristZ + landmrk.z
            # print(cx, cy)
            # print (ids, cx, cy)
            print(ids)
            x, y, z = int(cx), int(cy), int(cz)
            print('{}, {}, {}'.format(x, y, z))
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('MediaPipe Hands', image)
    # for i in df.columns:
    #     df[i] = ids, landmrk

    if cv2.waitKey(10) & 0xFF == 27:
      break
    elif cv2.waitKey(10) == ord('s'):
        # cv2.imwrite(f'my_pic_{index}.png', image) # s값을 누르면 사진을 저장한다.
        index = index + 1
        data = pd.DataFrame(data)
        for i in range(21):
            landmrk[i] = [x, y, z]
        data.to_csv('C:/workspace/rnn/data.csv', sep=',')

    
cap.release()

#python hand_tracking.py