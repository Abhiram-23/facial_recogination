import numpy as np
import cv2
import os

import img_rec as im
from traning import face_recoginizer

test_img = cv2.imread(r'C:\Users\pabhi\OneDrive\Desktop\Handwriting-master\Drowsiness detection\image.jpg')
faces_detected,gray_img = im.faceDetection(test_img)
print("face detected",faces_detected)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'C:\Users\pabhi\PycharmProjects\asish_py\trainingData.yml')
name={0:'abhi'}
for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+w,x:x+h]
    label,confidence = face_recoginizer.predict(roi_gray)
    print("label",label)
    print("Confidence",confidence)
    im.draw_rect(test_img,face)
    predict_name = name[label]
    im.put_text(test_img,predict_name,x,y)
resize_img =cv2.resize(test_img,(1000,700))
cv2.imshow("face detection",resize_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
