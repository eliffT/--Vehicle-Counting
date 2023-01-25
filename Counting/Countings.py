import cv2
import numpy as np


class Kordinat:
    def __init__(self,x,y):
         self.x=x
         self.y=y

class Sensor:
     def __init__(self,kordinat1,kordinat2,frame_weight,frame_lenght):
         self.kordinat1=kordinat1
         self.kordinat2=kordinat2
         self.frame_weight=frame_weight
         self.frame_lenght =frame_lenght
         self.mask=np.zeros((frame_lenght,frame_weight,1),np.uint8)
         self.full_mask_area=abs(self.kordinat2.x-self.kordinat1.x)*abs(self.kordinat2.y-self.kordinat1.y)
         cv2.rectangle(self.mask,(self.kordinat1.x,self.kordinat1.y),(self.kordinat2.x,self.kordinat2.y),(255),thickness=cv2.FILLED)
         self.stuation=False
         self.car_number_detected=0

Sensor1 = Sensor(Kordinat(400, 100), Kordinat(480, 150),600,350)    #En sağdaki şerit
Sensor2 = Sensor(Kordinat(70, 100), Kordinat(150, 150),600,350)     #En soldaki 1.şerit için
Sensor3 = Sensor(Kordinat(260, 100), Kordinat(340, 150), 600,350)   #Orta şerit

video=cv2.VideoCapture("Otobann.mp4")
subtrator=cv2.createBackgroundSubtractorMOG2()  #Arkaplan çıkarımı

kernel=np.ones((5,5),np.uint8)

while (1):
    ret,frame=video.read()
    frame=frame[250:600,0:600]
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),0)
    deleted_background=subtrator.apply(blur)
    opening_img=cv2.morphologyEx(deleted_background,cv2.MORPH_OPEN,kernel)
    closing_img = cv2.morphologyEx(opening_img,cv2.MORPH_CLOSE,kernel)
    
    cnts,_=cv2.findContours(closing_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)       #Konturların bulunması
    result=frame.copy()
    zeros_image=np.zeros((frame.shape[0],frame.shape[1],1),np.uint8)
    for cnt in cnts:
         x,y,w,h=cv2.boundingRect(cnt)                                              #Her bir kontur için rectangle oluşturulması
         if (w>50 and h>50 ):
            cv2.rectangle(result,(x,y),(x+w,y+h),(255,0,0),thickness=2)
            cv2.rectangle(zeros_image,(x,y),(x+w,y+h),(255),thickness=cv2.FILLED)   #Konturların beyaz ile doldurulması
            
    #Sensör1 oluşturuldu ve bu sensörün koordinatlarına göre kırmızı renkte bir dikdörtgen çizdirildi.        
    def Sensors(Sensor,result,zeros_image):
        #Sensor oluşturma
        cv2.rectangle(result,(Sensor.kordinat1.x,Sensor.kordinat1.y),(Sensor.kordinat2.x,Sensor.kordinat2.y),(255,0,0),thickness=cv2.FILLED)
         #Maskeleme Adımı
        mask_result=cv2.bitwise_and(zeros_image,zeros_image,mask=Sensor.mask)
        white_cell_number=np.sum(mask_result==255)
        sensor_rate=white_cell_number/Sensor.full_mask_area
        # Sensor oranına ve araç durumuna göre sayma adımları
        if (sensor_rate >= 0.55  and Sensor.stuation == False ):
            cv2.rectangle(result, (Sensor.kordinat1.x, Sensor.kordinat1.y), (Sensor.kordinat2.x, Sensor.kordinat2.y),
                       (214, 100, 120), thickness=cv2.FILLED)
            Sensor.stuation = True
          
        if (sensor_rate<0.55 and Sensor.stuation==True):
            cv2.rectangle(result, (Sensor.kordinat1.x, Sensor.kordinat1.y), (Sensor.kordinat2.x, Sensor.kordinat2.y),
                       (180,255, 255,), thickness=cv2.FILLED)
            Sensor.car_number_detected+=1
            Sensor.stuation = False
        else :
            cv2.rectangle(result, (Sensor.kordinat1.x, Sensor.kordinat1.y), (Sensor.kordinat2.x, Sensor.kordinat2.y),
                       (214, 100, 120), thickness=cv2.FILLED)
        cv2.imshow("mask_result ", mask_result)
        return Sensor.car_number_detected -1
    
    
    cv2.rectangle(result, (0,100), (600, 150),
                       (214, 100, 120), thickness=cv2.FILLED)
    
    t1 = Sensors(Sensor1,result,zeros_image)
    t2 = Sensors(Sensor2,result,zeros_image)
    t3 = Sensors(Sensor3,result,zeros_image)
    T = t1+t2+t3
    
    cv2.putText(result,str(T),(0,80),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),5)

    cv2.imshow("video", result)
    # cv2.imshow("Blur", blur)
    # cv2.imshow("Deleted background", deleted_background)
    # cv2.imshow("Opening", opening_img)
    # cv2.imshow("Closing", closing_img)
    k=cv2.waitKey(30) & 0xff
    if k == 27 :
        break

video.release()
cv2.destroyAllWindows()