import numpy as np
import math
""" 
assume the square is 100 pixels when view horizontally at 1m far
____camera____
       |
       |1.0m
       |
       |
____square____

______________
|     _       |
|    |_|100p  |720
|_____________|
        1080


"""
resolution = np.array([1080,720],dtype='int32')

focal_lenth = 0.005 # pixel
world_square = 2 # meter
# using variable in https://www.tutorialspoint.com/dip/perspective_transformation.htm
# y = YZ/f
# if we can set the pixel of square at 1 meter be correct, the function will run correctly
pixel_of_square_at_1m = world_square * 1 / focal_lenth # 1 is distance from camera
print("assumed pixel_of_square_at_1m:{},\n if this is wrong, the calculation will also be wrong, please change focal_length or world_square".format(pixel_of_square_at_1m))



def length(v):
  return math.sqrt(v[0]*v[0] + v[1]*v[1])

def real_position(vertices):
  if(len(vertices)!=4):
    print("given shape does not have equally 4 vertices")
    return
  print("assumed resolution:{}".format(resolution))
  print("assumed focal_lenth:{}".format(focal_lenth))
  print("assumed world_square:{}".format(world_square))
  vertices -= (resolution/2).astype('int32')
  for p in vertices:
    print(p)
  x = [p[0] for p in vertices]
  y = [p[1] for p in vertices]
  pixel_centroid = np.array([sum(x) / len(vertices), sum(y) / len(vertices)])
  pixel_centroid_distance = length(pixel_centroid)
  pixel_width = abs(length(vertices[0]-vertices[1]))
  pixel_height = abs(length(vertices[0]-vertices[3]))

  print("pixel_width:{}".format(pixel_width))
  print("pixel_distance:{}".format(pixel_centroid_distance))
  # assuming horizontal, so widht==height
  # Z = fy/Y
  distance_from_camera = pixel_width * focal_lenth / world_square
  # Y = fy/Z
  distance_from_point_below_camera_on_ground_plane = focal_lenth * pixel_centroid_distance / distance_from_camera

  print("distance_from_camera:{}".format(distance_from_camera))
  print("distance_from_point_below_camera_on_ground_plane:{}".format(distance_from_point_below_camera_on_ground_plane))
  world_ground_plane_position = pixel_centroid * (distance_from_point_below_camera_on_ground_plane  / pixel_centroid_distance)
  world_position = np.array([world_ground_plane_position[0],-distance_from_camera,world_ground_plane_position[1]])
  print(world_position)
  return world_position


if(__name__=="__main__"):
  (x,y) = (500,600)
  size = 200

  x1 = x - size / 2
  x2 = x + size / 2
  y1 = y - size / 2
  y2 = y + size / 2 
  #vertices = np.array([[500,300],[600,300],[600,400],[500,400]],dtype='int32')
  vertices = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y1]],dtype='int32')
  real_position(vertices)