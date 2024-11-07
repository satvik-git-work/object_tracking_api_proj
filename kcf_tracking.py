import cv2
import numpy as np

tracker=cv2.TrackerKCF_create()

vc=cv2.VideoCapture(0)
ret, frame=vc.read()

roi=cv2.selectROI(frame,False)

tracker.init(frame,roi)

while True:
    ret, fr=vc.read()
    success,roi=tracker.update(fr)
    (x,y,w,h)=tuple(map(int,roi))

    if success:
        cv2.rectangle(fr,(x,y),(x+w,y+h),(200,150,100),2)
    else:
        break    

    cv2.imshow("kernelized correlation filters",fr)

    if cv2.waitKey(1) & 0xFF ==27:
        break


vc.release()
cv2.destroyAllWindows()    