import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pygame
count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
count7=0
pygame.init()
pygame.mixer.init()
model = load_model("emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi)[0]
            label = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if label=="Happy":
                print("Happy:")
                if(count1%25==0):
                    pygame.mixer.music.load("happy.mp3")
                    pygame.mixer.music.play(-1,fade_ms=10)
                count1+=1
            if label=="Sad":
                print("Sad")
                if(count2%25==0):
                    pygame.mixer.music.load("sad.mp3")
                    pygame.mixer.music.play(0,fade_ms=10)
                count2+=1
            if label=="Surprise":
                print("Surprise")
                if(count3%25==0):
                    pygame.mixer.music.load("surprise.mp3")
                    pygame.mixer.music.play(-1,fade_ms=10)
                count3+=1
            if label=="Angry":
                print("Angry")
                if(count4%25==0):
                    pygame.mixer.music.load("angry.mp3")
                    pygame.mixer.music.play(-1,fade_ms=10)
                count4+=1
            if label=="Fear":
                print("Fear")
                if(count5%25==0):
                    pygame.mixer.music.load("fear.mp3")
                    pygame.mixer.music.play(-1,fade_ms=10)
                count5+=1
            if label=="Neutral":
                print("Neutral")
                if(count6%50==0):
                    pygame.mixer.music.load("neutral.mp3")
                    pygame.mixer.music.play(-1,fade_ms=10)
                count6+=1
            if label=="Disgust":
                print("Disgust")
                if(count7%25==0):
                    pygame.mixer.music.load("disgust.mp3")
                    pygame.mixer.music.play(-1,fade_ms=10)
                count7+=1

    cv2.imshow("Facial Emotion Recognition", frame)

    if cv2.waitKey(10)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
