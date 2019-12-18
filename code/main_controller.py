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
    box = box_detection.find_parr_in_frame(frame)
    if(box is not None):
        tasks.append(Tasks(frame,box))
        print("new task is added, there are {0} tasks".format(len(tasks)))
        cv2.drawContours(ori_image, [box], -1,(0,255,0),2) #[box] or box are ok
    else:
        print("no task is added, {} tasks in queue".format(len(tasks)))
    if(not usingPiCamera):
        cv2.imshow("Original", ori_image)
    key = cv2.waitKey(1)

def repeat_do_tasks():
    while(not programEnd):
        if(len(tasks)>0):
            print("do new tasks, still have {} tasks left".format(len(tasks)))
            current_task = tasks.pop()
            results.append(Results(current_task,recognition.apply_recognition(current_task.frame,current_task.box, not usingPiCamera)))
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
