import dlib
import cv2
from imutils import face_utils
import numpy as np
import argparse
import imutils

def run(image):
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # load the input image, resize it, and convert it to grayscale
        #image = cv2.imread(image)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = detector(gray, 1)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
                # convert the landmark (x, y)-coordinates to a NumPy array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # loop over the face parts individually
                for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                        # clone the original image so we can draw on it, then
                        # display the name of the face part on the image
                        clone = image.copy()
                        # loop over the subset of facial landmarks, drawing the
                        # specific face part
                        xMax = 0
                        yMax = 0
                        for (x, y) in shape[i:j]:
                                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                                if x > xMax:
                                        xMax = x
                                if y > yMax:
                                        yMax = y

                        # extract the ROI of the face region as a separate image
                        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                        differenceY = yMax-y
                        differenceX = xMax-x
                        x -= differenceX/4
                        w += differenceX/2
                        y -= differenceY/4
                        h += differenceY/2
                        roi = image[y:y + h, x:x + w]
                        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
                        cv2.imwrite("ROI.png", roi)

                        # show the particular face part
                        #cv2.imshow("ROI", roi)
                        #cv2.imshow("Image", clone)
                        #cv2.waitKey(0)
                        break

