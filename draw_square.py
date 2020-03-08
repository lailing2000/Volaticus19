import numpy as np
import math
import cv2

if(__name__=="__main__"):
  (x,y) = (500,600)
  size = 200

  x1 = x - size / 2
  x2 = x + size / 2
  y1 = y - size / 2
  y2 = y + size / 2 
  #vertices = np.array([[500,300],[600,300],[600,400],[500,400]],dtype='int32')
  vertices = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y1]],dtype='int32')
  frame = np.zeros((1080,720))
  
  cv2.imshow("Original", frame)
  print("finsih")
  key = cv2.waitKey(1)
