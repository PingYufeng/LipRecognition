# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
from PIL import Image
from resizeimage import resizeimage
import os
from random import randint
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

word = "hello"
ogVideo = "/Users/sameer/Desktop/Python/FaceRecognition/trainVideos/Anand.mp4"

t1 = 0
t2 = 1000

ffmpeg_extract_subclip(ogVideo, t1, t2, targetname="splitVideo.mp4")

#create parent directory
uniqueID = randint(0, 100000)
parent_file_path = "/Users/sameer/Desktop/Python/FaceRecognition/" + word + "/"
directory = os.path.dirname(parent_file_path)
if not os.path.exists(directory):
    os.makedirs(directory)

#create folder with unique identification
id_file_path = "/Users/sameer/Desktop/Python/FaceRecognition/" + word + "/" + str(uniqueID) + "/"
print(id_file_path)
directory_unique = os.path.dirname(id_file_path)
if not os.path.exists(directory_unique):
    os.makedirs(directory_unique)
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vidcap = cv2.VideoCapture(video)
success,image = vidcap.read()
time.sleep(2.0)
count = 0
externalCount = 0
success = True

# loop over the frames from the video stream
while success:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	ret, frame = vidcap.read()

        if frame is None:
                break
	
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

                shape = shape[48:len(shape)]
		
		#print(shape)
                for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                        # loop over the (x, y)-coordinates for the facial landmarks
                        # and draw them on the image
                        xMin = 10000
                        xMax = 0
                        yMin = 10000
                        yMax = 0
                        for (x, y) in shape:
                                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                                if x < xMin:
                                        xMin = x
                                if y < yMin:
                                        yMin = y
                                if x > xMax:
                                        xMax = x
                                if y > yMax:
                                        yMax = y
	  
                        # show the frame
                        cv2.imshow("Frame", frame)
                        key = cv2.waitKey(1) & 0xFF
                        
                        if externalCount % 2:   
                             success,image = vidcap.read()
                             #print('Read a new frame: ', success)
                             if success:
                                #(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                                x = xMin
                                w = xMax - xMin
                                y = yMin
                                h = yMax - yMin
                                #print count
                                #print 'x:',x
                                #print 'w:', w
                                #print 'y: ',y
                                #print 'h: ', h
                                #cv2.imwrite('image.png', image)
                                roi = image[y*2:(y+h+h+h)*2, x*2:(x+w+w+w)*2]
                                #roi = image
                                #print(roi.shape)
                                #print image.shape[0]
                                #print image.shape[1]
                                #cv2.imwrite('roi.png', roi)
                                roi = imutils.resize(roi, width=400, inter=cv2.INTER_CUBIC)
                                writeDirectory = "/Users/sameer/Desktop/Python/FaceRecognition" + "/" + word + "/" + str(uniqueID)
                                resizeWriteDir = "/Users/sameer/Desktop/Python/FaceRecognition" + "/" + word + "/" + str(uniqueID)
                                cv2.imwrite(os.path.join(writeDirectory, "ROI.png"), roi)
                                with open(os.path.join(writeDirectory, "ROI.png"), 'r+b') as f:
                                        with Image.open(f) as image:
                                           cover = resizeimage.resize_cover(image, [400, 225], validate=False)
                                           cover.save(os.path.join(writeDirectory, '%d.png' % count), image.format)
                                #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
                             count += 1
                        externalCount += 1

                        break

 
# do a bit of cleanup
cv2.destroyAllWindows()
vidcap.release()
