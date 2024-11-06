import mediapipe as mp
import cv2
import numpy as np
import pickle
import pandas as pd 
import csv

mp_drawing  = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

with open('Hand_test_new.pkl','rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

        while cap.isOpened():

             re , frame = cap.read()

             image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
             image = cv2.flip(image,1)
             image.flags.writeable = False

             results = hands.process(image)
             image.flags.writeable = True

             #print(results.multi_hand_landmarks)

             if results.multi_hand_landmarks:
             
                 for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS)
             

             image =cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

             try:
                 for landmark in results.multi_hand_landmarks:
                     pose = landmark.landmark
                     pose_row = list(np.array([[landmark.x , landmark.y , landmark.z] for landmark in pose]).flatten())
                    

                 row= pose_row
                 X = pd.DataFrame([row])
                 body_language_class = model.predict(X)[0]
                 body_language_prob = model.predict_proba(X)[0]
            # print(body_language_class , body_language_prob)

            # grab ear coords
                 coords =(50,50)

                 cv2.rectangle(image,
                 (coords[0],coords[1]+5) , 
                 (coords[0]+len(body_language_class)*20,coords[1]-30), (245 , 117 , 16),-1)

                 cv2.putText(image , body_language_class , coords ,
                 cv2.FONT_HERSHEY_SIMPLEX , 1 , (255,255,255),2 , cv2.LINE_AA)

                

             except:
                 pass




             cv2.imshow('image',image)

             if cv2.waitKey(10) & 0xff == ord('q'):
                 break


cap.release()
cv2.destroyAllWindows()