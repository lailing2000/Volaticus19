import cv2, time
import numpy as np
import imutils
import math
import pytesseract
import threading
import keyboard
import recognition
import box_detection

usingPiCamera = False
try:
    from picamera.array import PiRGBArray
    from picamera import PiCamera
    usingPiCamera = True
    print("using py camera")
except:
	print("not using py camera")


class Tasks:
    def __init__(self, frame, box, color, camera_position, camera_orientation, flight_time):
        self.frame = frame
        self.box = box
        self.color = color
        self.camera_position = camera_position
        self.camera_orientation = camera_orientation
        self.flight_time = flight_time
    def __str__(self):
        return "flight_time:{}, color:{}".format(self.flight_time, self.color)
    def __repr__(self):
        return self.__str__

class Results:
    def __init__(self, task, result):
        self.task = task
        self.result = result


start_time = time.time()

def detect_square(frame):
    ori_image = frame
    box, color = box_detection.find_square(frame)
    if(box is not None):
        location = (1,2,3)
        rotation = (1,2,3)
        flight_time = time.time() - start_time
        tasks.append(Tasks(frame, box, color, location, rotation, flight_time))
        
        print("new task is added, there are {0} tasks".format(len(tasks)))
        cv2.drawContours(ori_image, [box], -1,(0,255,0),2) #[box] or box are ok
        
    if(not usingPiCamera):
        cv2.imshow("Original", ori_image)
    key = cv2.waitKey(1)

def read_square():
    while(not programEnd):
        if(len(tasks)>0):
            print("do new tasks, still have {} tasks left".format(len(tasks)))
            current_task = tasks.pop()
            print("task info:{}".format(current_task))
            results.append(Results(current_task,recognition.recognize(current_task.frame,current_task.box, not usingPiCamera)))
        time.sleep(.1)
def detect():
    if(usingPiCamera):
        camera = PiCamera()
        camera.resolution = (640, 480)
        camera.framerate = 10
        rawCapture = PiRGBArray(camera, size=(640, 480))
        time.sleep(0.1)
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            frame = frame.array
            detect_square(frame)
            rawCapture.truncate(0)
            if(programEnd):
                break
    else:
        cap = cv2.VideoCapture(0)
        while(True):
            ret, frame = cap.read()
            detect_square(frame)
            if(programEnd):
                break

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
    fbThread=threading.Thread(name="find box",target=detect)
    fbThread.start()
    recThread=threading.Thread(name="reconition",target=read_square)
    recThread.start()
    inputThread = threading.Thread(name="get input",target=get_input)
    inputThread.start()
    fbThread.join()
    recThread.join()
    inputThread.join()
    for result in results:
        print(result.result)

main_loop()
