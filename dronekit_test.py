

from dronekit import connect
import time

connection_string = "/dev/serial0"
print('connecting to drone')


vehicle = connect('/dev/serial0',baud=921600,rate=10) # 921600 57600
vehicle.wait_ready(True, timeout=300)
print('connected')
print(vehicle.parameters)
print(vehicle.location.local_frame)
print(vehicle.gps_0)
print(vehicle.battery)
print(vehicle.mode)

vehicle.simple_takeoff()
while(True):
  time.sleep(1)
  print(vehicle.gps_0)
  print(vehicle.attitude.pitch)
  print(vehicle.attitude.yaw)

vehicle.close()
time.sleep(1)
