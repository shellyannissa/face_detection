import cv2
import time
import uuid
import os

IMAGES_PATH='./sample_data'
if not os.path.exists(IMAGES_PATH):
  os.mkdir(IMAGES_PATH)


cap=cv2.VideoCapture(0)
for imgnum in range(60):
  print('Collecting image {}'.format(imgnum))
  ret,frame=cap.read()
  if (not cap.isOpened()):
    print("ERROR! Unable to open camera")


  imgname=os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
  cv2.imwrite(imgname,frame)
  cv2.imshow('frame',frame)
  time.sleep(0.5)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()