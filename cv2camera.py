import cv2
import numpy as np
from tensorflow import keras
import simpleaudio as sa

# Load the resources
model = keras.models.load_model('model')
haarcascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
alarm = sa.WaveObject.from_wave_file('resources/alarm.wav')

def process_frame(frame):
    # Detect faces with haarcascade
    faces = haarcascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=7)
    # If a face is detected predict if the person wear a mask and return the frame
    if len(faces) != 0:
        for (x, y, w, h) in faces:
            image_predict = frame[y-75:y+h+75,x-35:x+w+35]
            image_predict = cv2.resize(image_predict, (96, 96))
            image_predict = image_predict.reshape(-1, 96, 96, 3).astype('float')/255
            pred = model.predict_classes(image_predict)

        return pred[0], frame
    return 2, frame

cam = cv2.VideoCapture(0)

# Load the emoji image
smiley = cv2.imread('resources/smiley.jpg')
smiley = cv2.resize(smiley, (100, 100))

timer = 20

while True:
    ret, frame = cam.read()
    if ret == True:
        # Make the prediction on the image
        pred, image = process_frame(frame)
        # If the result is 'No mask', play an alarm song for 20 frames
        if pred == 0:
            if timer == 20:
                play = alarm.play()
                timer -= 1
            elif timer == 0:
                timer = 20
            else:
                timer -= 1
        
        # If the result is 'Mask', display the emoji in the top left corner and stop the alarm     
        elif pred == 1:
            try:
                play.stop()
            except:
                pass
            timer = 20
            frame[50:150, 50:150] = smiley
        
        # If no face is detected, continue
        elif pred == 2:
            continue

    cv2.imshow("test", image)

    k = cv2.waitKey(1)
    if k%256 == 27:
        break

cam.release()
cv2.destroyAllWindows()