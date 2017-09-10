import cv2
import os
from random import randint
import ImageResizer


word = "the"

uniqueID = randint(0, 100000)
parent_file_path = "/Users/sameer/Desktop/Python/FaceRecognition/" + word + "/"
directory = os.path.dirname(parent_file_path)
if not os.path.exists(directory):
    os.makedirs(directory)

id_file_path = "/Users/sameer/Desktop/Python/FaceRecognition/" + word + "/" + str(uniqueID) + "/"
print(id_file_path)
directory_unique = os.path.dirname(id_file_path)
if not os.path.exists(directory_unique):
    print("hello")
    os.makedirs(directory_unique)

vidcap = cv2.VideoCapture('TabethaBoyajianThemostmysteriousstarintheuniverse.mp4')
success,image = vidcap.read()
count = 0
externalCount = 0
success = True
while success:
  if externalCount % 3:   
     success,image = vidcap.read()
     print('Read a new frame: ', success)
     if success:
         #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
         ImageResizer.run(image, count)
     count += 1
  externalCount += 1
          
