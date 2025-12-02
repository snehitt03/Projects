import cv2
import numpy as np
from sklearn.svm import SVC
import joblib
import pywt

def apply_wavelet_transform(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=np.float32(image)
    image/=255;
    coeffs=pywt.wavedec2(image,wavelet='db1',level=5)

    ##cA,(cH,cV,cD)=coeffs
    ##return cA
    coeffs_H=list(coeffs)
    coeffs_H[0]*=0;
    
    image_H=pywt.waverec2(coeffs_H,wavelet='db1')
    image_H[0]*=255;
    image_H=np.uint8(image_H)
    return image_H

model=joblib.load('saved_model.pkl')
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#img=cv2.imread("nm.jpg")
# ! @ () 3
 

names=['cristiano_ronaldo','elon_musk','john_cena','leonel_messi','narendra_modi','neeraj_chopra','putin','rameshbabu_praggnanandhaa','sunil_chettri','thalapathy_vijay','virat_kohli']

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    if not ret:
        break
    
    
    #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #faces=face_cascade.detectMultiScale(gray,1.1,4,minSize=(20,20))
    faces=face_cascade.detectMultiScale(frame,1.1,4,minSize=(20,20))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        face_roi=frame[y:y+h,x:x+w]
        face_resized=cv2.resize(face_roi,(32,32))
        orig_face_resized=face_resized #scalled_raw_image
        #wavelet transformed
        face_transformed=apply_wavelet_transform(face_resized)
        #resize the transformed image
        scaled_face_t=cv2.resize(face_transformed,(32,32)) #scalled_image
        #vertically stack both
        ###face_input=face_transformed.flatten().reshape(1,-1)

        face_input=np.vstack((orig_face_resized.reshape(32*32*3,1),scaled_face_t.reshape(32*32,1)))
        

        prediction=model.predict(face_input)
        name=names[int(prediction[0])]

        cv2.putText(frame,name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
    
    cv2.imshow('real-time-recog',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()