import mediapipe as mp
import cv2
import csv
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():

         re , frame = cap.read()

         image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
         image = cv2.flip(image,1)
         image.flags.writeable = False

         results = hands.process(image)
         image.flags.writeable = True

         print(results.multi_hand_landmarks)

         if results.multi_hand_landmarks:
             
             for num, hand in enumerate(results.multi_hand_landmarks):
                 mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS)
             

         image =cv2.cvtColor(image,cv2.COLOR_RGB2BGR)



         cv2.imshow('image',image)

         if cv2.waitKey(10) & 0xff == ord('q'):
             break


cap.release()
cv2.destroyAllWindows()

num_coords = 21

landmarks = ['class']
for val in range(1,num_coords+1):
    landmarks += ['x{}'.format(val) ,'y{}'.format(val),'z{}'.format(val)]

with open('handmarks.csv' , mode='w',newline='') as f:
    csv_writer = csv.writer(f , delimiter=',',quotechar='"',quoting = csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)

userinput=input("Enter the Value to be trained or n to Quit \n")

while userinput !='n':
    class_name = userinput
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
                 row.insert(0 , class_name)

                 with open('handmarks.csv' , mode='a',newline='') as f:
                    csv_writer = csv.writer(f , delimiter=',',quotechar='"',quoting = csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)

             except:
                 pass




             cv2.imshow('image',image)

             if cv2.waitKey(10) & 0xff == ord('q'):
                 break


    cap.release()
    cv2.destroyAllWindows()
    userinput=input("Enter the Value to be trained or Press n to Quit \n")
