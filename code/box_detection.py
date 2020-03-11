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
  
def find_square(frame):
    image = cv2.GaussianBlur(frame,(15,15),0)
    #https://stackoverflow.com/questions/31460267/python-opencv-color-tracking
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cut = 20
    boxes = []
    boxes_size = []
    colors = []
    white_cnts = detect_white_squares(hsv)
    
    if(len(white_cnts) == 0):
        return None, None

    for c in range(0,cut):
        current_color = int(180 / cut * c)
        current_largest_box = detect_colored_square(hsv,current_color, white_cnts)
        if(current_largest_box is not None):
            boxes.append(current_largest_box)
            boxes_size.append(get_box_size(current_largest_box))
            colors.append(current_color)
    if(len(boxes) > 0):
        largest = np.argmax(boxes_size)
        return boxes[largest], colors[largest]
    return None, None

def detect_white_squares(hsv):
    smin = 0
    smax = 100
    vmin = 150
    vmax = 255
    lower_bound = np.array([0, smin, vmin]) 
    upper_bound = np.array([180, smax, vmax])
    mask = cv2.inRange(hsv, lower_bound , upper_bound)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return get_squares(cnts)



def inside_white_square(cnt, white_squares_cnt):
    center0 = np.mean(cnt, axis=(0,1))
    size0 = get_box_size(cnt)
    for w_square in white_squares_cnt:
        center = np.mean(w_square, axis=(0,1))
        if(np.linalg.norm(center0-center)<3 and get_box_size(w_square) > size0):
            # print("dist:{}".format(np.linalg.norm(center0-center)))
            # print("white size:{}".format(find_box_size(w_square) ))
            return True
    return False

def get_cnt(hsv, color):
    '''return the cnts after applying the color filter to hsv image'''
    sensitivity = 10
    smin = 155
    smax = 255
    vmin = 155
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
    return cnts

def detect_colored_square(hsv, color, white_cnts):
    cnts = get_cnt(hsv, color)
    return get_largest_square(cnts, white_cnts)


def get_squares(cnts):
    '''filter cnts where they are square'''
    boxes = []
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
            if(r1 > 0.8 and r1 < 1.2 and r2 > 0.8 and r2 < 1.2):
                size = get_box_size(approx)
                if(size>140625):
                    boxes.append(approx)
    return boxes

def get_largest_square(cnts, white_cnts):
    '''
    get the largest square from a list of cnts, 
        where these squares have to be inside any white squares
    '''
    boxes = []
    box_sizes = []
    for cnt in cnts:
        cnt = cv2.convexHull(cnt)
        peri = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,0.1 * peri,True)
        if(len(approx) == 4):
            size = get_box_size(approx)
            if(size > 8789):
                if(inside_white_square(approx, white_cnts)):
                    boxes.append(approx)
                    box_sizes.append(get_box_size(approx))
    if(len(boxes)==0):
        return None
    index = np.argmax(box_sizes)
    largest_size = 10 #shortest side have to be at least 10 width
    if(box_sizes[index]>largest_size):
        print(box_sizes)
        return boxes[index]
    return None


def get_box_size(box):
    '''return area*area of the box'''
    # the reason of area*area is that it consider all sides/areas combination
    # not using sqrt to increase performance
    e1 = length((box[1] - box[0])[0])
    e2 = length((box[2] - box[3])[0])
    e3 = length((box[3] - box[0])[0])
    e4 = length((box[2] - box[1])[0])
    return e1*e2*e3*e4

def is_blur(frame, threashold = 100):
    print(int(cv2.Laplacian(frame,cv2.CV_64F).var()))
    return (cv2.Laplacian(frame,cv2.CV_64F).var() < threashold)


