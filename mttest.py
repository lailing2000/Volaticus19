import threading
import random
import time
import keyboard

def generateRandomTask(id=0):
    a = random.randrange(0,100)
    b = random.randrange(0,100)
    print("new task is {0} * {1}".format(a,b))
    tasks.append(Task(a,b))

def repeatGenerate(id=0):
    while(not programEnd):
        generateRandomTask(id)

class Task:
    a = 0
    b = 0
    result = 0
    def __init__(self,a,b):
        self.a = a
        self.b = b

programEnd = False
tasks = []
results = []
def doTasks(task):
    print("doing task + {0}*{1} at time {2}".format(task.a,task.b,time.time()))
    time.sleep(random.randrange(1.,5.))
    task.result = task.a * task.b
    results.append(task)
    print("finished doing task + {0}*{1}".format(task.a,task.b))

def repeatDoTasks():
    while(not programEnd):
        print("repeatDoTasks")
        if(len(tasks)>0):
            doTasks(tasks.pop())
        time.sleep(.5)

def mainThread():
    global programEnd
    while(True):
        #print("looping main thread")
        #time.sleep(.1)
        if(keyboard.is_pressed('q')):
            programEnd = True
            break
        if(keyboard.is_pressed('w')):
            generateRandomTask()
    programEnd = True

lock = threading.Lock()

generateRandomTask()
d2=threading.Thread(name="generate",target=repeatDoTasks)
d2.start()
d1=threading.Thread(name="generate",target=mainThread)
d1.start()

d1.join()
d2.join()
print("finish")