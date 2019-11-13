import cv2, time
import numpy as np
import imutils
import math
import pytesseract
import threading
import keyboard

usingPiCamera = False
try:
    from picamera.array import PiRGBArray
    from picamera import PiCamera
    usingPiCamera = True
    print("using py camera")
except:
	print("not using py camera")
# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect
def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped
	
def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
  
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
    green = 60
    blue = 120
    yellow = 30
    #https://stackoverflow.com/questions/31460267/python-opencv-color-tracking
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # return largest_parr(hsv,0)
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
    if(sensitivity > color):
	    lower_red_0 = np.array([0, smin, vmin]) 
	    upper_red_0 = np.array([color + sensitivity, smax, vmax])
	    lower_red_1 = np.array([180 - sensitivity + color, smin, vmin]) 
	    upper_red_1 = np.array([180, smax, vmax])
	
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
    # lowAngle = math.pi / 2 / 1.05
    # upAngle = math.pi / 2 * 1.05
    largestSide = 10 #shortest side have to be at least 10 width
    box = None
    for cnt in cnts:
        cnt = cv2.convexHull(cnt)
        peri = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,0.1 * peri,True)
        if(len(approx) == 4):
            isSquare = False
            e1 = approx[1] - approx[0]
            e2 = approx[2] - approx[3]
            
            e3 = approx[3] - approx[0]
            e4 = approx[2] - approx[1]
            
            r1 = length(e1[0])/length(e2[0])
            
            r2 = length(e3[0])/length(e4[0])
            if(r1 > 0.9 and r1 < 1.1 and r2 > 0.9 and r2 < 1.1):
            	isSquare = True
            
            if(isSquare):
                
                shortest_side = find_box_size(approx)
                if(shortest_side>largestSide):
                    largestSide = shortest_side
                    box = approx
    return box

def find_box_size(box):
    
    v1 = box[1] - box[0]
    v2 = box[2] - box[0]
    v3 = box[3] - box[0]
    
    l1 = length(v1[0])
    l2 = length(v2[0])
    l3 = length(v3[0])
    shortest_side = l1
    if(l2<shortest_side):
        shortest_side = l2
    if(l3<shortest_side):
        shortest_side = l3

    return shortest_side



def apply_recognition(frame, box):
    box = box[:,0]
    cropped_frame = four_point_transform(frame,box)
    # invert map white to black, red to blue
    inverted_cropped_frame = np.invert(cropped_frame)
    for i in range(0,4):
        print(i)
        rotated_inverted_cropped_frame = rotateImage(inverted_cropped_frame,90*i)
        #text = pytesseract.image_to_string(b, config="-l eng --oem 1 --psm 10")
        minConf = 70
        data = pytesseract.image_to_data(rotated_inverted_cropped_frame, config="-l eng --oem 1 --psm 10", output_type=pytesseract.Output.DICT)
        # print(str(data['conf'][len(data['conf'])-1])+","+str(data['text'][len(data['text'])-1]))
        # it is still not good at recognising some characters such as C
        conf = data['conf'][len(data['conf'])-1]
        text = str(data['text'][len(data['text'])-1])
        print("result : conf="+str(conf)+", text="+text)
        if conf=='-1' or conf < minConf or len(text) != 1:
            continue
        char = ord(text)
        if not((char >= 48 and char <=57) or (char >=65 and char <=90)):
            continue
        print(text)
        return text

class Tasks:
    frame = None
    box = None
    def __init__(self, frame, box):
        self.frame = frame
        self.box = box

class Results:
    task = None
    result = None

    def __init__(self, task, result):
        self.task = task
        self.result = result


def recognize_frame(frame):
    ori_image = frame
    box = find_parr_in_frame(frame)
    if(box is not None):
        tasks.append(Tasks(frame,box))
        print("new task is added, there are {0} tasks".format(len(tasks)))
        cv2.drawContours(ori_image, [box], -1,(0,255,0),2) #[box] or box are ok
        # apply_recognition(frame, box)
    else:
        print("no task is added, {} tasks in queue".format(len(tasks)))
    if(not usingPiCamera):
        #cv2.imshow("Capturing", mask)
        cv2.imshow("Original", ori_image)
    key = cv2.waitKey(1)

def repeat_do_tasks():
    while(not programEnd):
        if(len(tasks)>0):
            print("do new tasks, still have {} tasks left".format(len(tasks)))
            current_task = tasks.pop(0)
            results.append(Results(current_task,apply_recognition(current_task.frame,current_task.box)))
        time.sleep(.1)

def get_input():
    global programEnd
    while(not programEnd):
        x = input()
        if(x == 'q'):
            programEnd = True


programEnd = False
tasks = []
results = []
def main_loop():
    global programEnd 
    recThread=threading.Thread(name="reconition",target=repeat_do_tasks)
    recThread.start()
    inputThread = threading.Thread(name="get input",target=get_input)
    inputThread.start()
    if(usingPiCamera):
        camera = PiCamera()
        camera.resolution = (640, 480)
        camera.framerate = 10
        rawCapture = PiRGBArray(camera, size=(640, 480))
        time.sleep(0.1)
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            frame = frame.array
            recognize_frame(frame)
            rawCapture.truncate(0)
            if(programEnd):
                break
    else:
        cap = cv2.VideoCapture(0)
        while(True):
            ret, frame = cap.read()
            recognize_frame(frame)
            if(programEnd):
                break
    recThread.join()
    inputThread.join()
    for result in results:
        print(result.result)
main_loop()
