import cv2, time
import numpy as np
import imutils
import math

  
def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(v[0]*v[0] + v[1]*v[1])

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def area(v1, v2):
  return length(np.cross(v1, v2))
  
def find_parr_in_frame(frame):
    image = cv2.GaussianBlur(frame,(15,15),0)
    #https://stackoverflow.com/questions/31460267/python-opencv-color-tracking
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # return largest_parr(hsv,0)# only red
    cut = 20
    boxes = []
    for c in range(0,cut):
        current_color = int(180 / cut * c)
        current_largest_box = largest_parr(hsv,current_color)
        if(current_largest_box is not None):
            boxes.append(current_largest_box)
    if(len(boxes) > 0):
        largest_box_size = 0
        largest_box = None
        for box in boxes:
            current_size = find_box_size(box)
            if(current_size > largest_box_size):
                largest_box_size = current_size
                largest_box = box
        return largest_box




def largest_parr(hsv, color):

    sensitivity = 10
    smin = 100
    smax = 255
    vmin = 100
    vmax = 255
    if(color - sensitivity < 0):
	    lower_red_0 = np.array([0, smin, vmin]) 
	    upper_red_0 = np.array([color + sensitivity, smax, vmax])
	    lower_red_1 = np.array([180 - sensitivity + color, smin, vmin]) 
	    upper_red_1 = np.array([180, smax, vmax])
	
	    mask_0 = cv2.inRange(hsv, lower_red_0 , upper_red_0)
	    mask_1 = cv2.inRange(hsv, lower_red_1 , upper_red_1 )
	    mask = cv2.bitwise_or(mask_0, mask_1)
    elif(color + sensitivity > 180):
	    lower_red_0 = np.array([color - sensitivity, smin, vmin]) 
	    upper_red_0 = np.array([180, smax, vmax])
	    lower_red_1 = np.array([0, smin, vmin]) 
	    upper_red_1 = np.array([color + sensitivity - 180,smax, vmax])
	
	    mask_0 = cv2.inRange(hsv, lower_red_0 , upper_red_0)
	    mask_1 = cv2.inRange(hsv, lower_red_1 , upper_red_1 )
	    mask = cv2.bitwise_or(mask_0, mask_1)
    else:
	    lower_red_0 = np.array([color - sensitivity, smin, vmin]) 
	    upper_red_0 = np.array([color + sensitivity, smax, vmax])
	    mask = cv2.inRange(hsv, lower_red_0 , upper_red_0)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    boxes = []
    box_sizes = []
    for cnt in cnts:
        cnt = cv2.convexHull(cnt)
        peri = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,0.1 * peri,True)
        if(len(approx) == 4):
            e1 = approx[1] - approx[0]
            e2 = approx[2] - approx[3]
            e3 = approx[3] - approx[0]
            e4 = approx[2] - approx[1]
            r1 = length(e1[0])/length(e2[0])
            r2 = length(e3[0])/length(e4[0])
            # comparing ratio of length of opposite side
            # if it is parallelogram, the opposite side should have equal side
            if(r1 > 0.9 and r1 < 1.1 and r2 > 0.9 and r2 < 1.1):
                boxes.append(approx)
                box_sizes.append(find_box_size(approx))
    if(len(boxes)==0):
        return None
    index = np.argmax(box_sizes)
    largest_size = 10 #shortest side have to be at least 10 width
    if(box_sizes[index]>largest_size):
        return boxes[index]

    return None

def find_box_size(box):
    '''
    return the smallest side
    '''
    v1 = box[1] - box[0]
    v2 = box[2] - box[0]
    v3 = box[3] - box[0]
    l1 = length(v1[0])
    l2 = length(v2[0])
    l3 = length(v3[0])
    return np.min([l1,l2,l3])

def isBlur(frame, threashold = 100):
    print(int(cv2.Laplacian(frame,cv2.CV_64F).var()))
    return (cv2.Laplacian(frame,cv2.CV_64F).var() < threashold)


