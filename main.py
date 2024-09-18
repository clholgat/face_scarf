import argparse
import time
import numpy as np
# opencv 4.1.2 to read images
import cv2

# used for accessing url to download files
import urllib.request as urlreq

# used to access local directory
import os

# used to plot our images
import matplotlib.pyplot as plt

# used to change image size
from pylab import rcParams

from PIL import Image
import math

# For cropping [left, top, right, bottom]
PADDING = [10, 55, 10, 50]


print("Building you a face")

def display_rect(img, x, y, w, l):
    cv2.rectangle(img,(x,y),(x+w, y+l),(255, 255, 255), 2)

    plt.axis("off")
    plt.imshow(img)
    plt.title('Face Detection')
    plt.show()

def crop_to_face(img, face):
    (x,y,w,l) = face

    x = x - PADDING[0]
    y = y - PADDING[1]
    w = w + PADDING[2] + PADDING[0]
    l = l + PADDING[3] + PADDING[1]

    return (img[y:(y+l), x:(x+w)] , (face[0] - x, face[1] - y, face[2], face[3]))

def find_face_and_features(img):
    # Model should be local
    haarcascade = "haarcascade_frontalface_alt2.xml"

    # create an instance of the Face Detection Cascade Classifier
    detector = cv2.CascadeClassifier(haarcascade)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = detector.detectMultiScale(imgGray)

    if len(faces) > 10:
        print("MULTIPLE FACES")

        # Display faces for debugging purposes
        for face in faces:
            # save the coordinates in x, y, w, d variables
            (x,y,w,l) = face
            # Draw a white coloured rectangle around each face using the face's coordinates
            # on the "image_template" with the thickness of 2 
            cv2.rectangle(imgGray,(x,y),(x+w, y+l),(255, 255, 255), 2)

        plt.axis("off")
        plt.imshow(imgGray)
        plt.title('Face Detection')
        plt.show()
        #return None
    
    face = faces[0]

    #Find features

    lbfmodel = "lbfmodel.yaml"

    # create an instance of the Facial landmark Detector with the model
    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(lbfmodel)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect landmarks on "image_gray"
    _, landmarks = landmark_detector.fit(imgGray, np.array([face]))

    for landmark in landmarks:
        eyes = [landmark[0][36], landmark[0][45]]
        for x,y in eyes:
            # display landmarks on "image_cropped"
            # with white colour in BGR and thickness 1
            cv2.circle(imgGray, (int(x), int(y)), 1, (255, 255, 255), 10)
    plt.axis("off")
    plt.imshow(imgGray)
    plt.show()

    return (face, landmark[0])

def deg_off_level(landmarks):
    # Corners of the eyes
    leftX, leftY = landmarks[36]
    rightX, rightY = landmarks[45]

    # Same y values means face is level
    if leftY == rightY:
        return 0
    
    hyp = math.sqrt((rightX - leftX) ** 2 + (rightY - leftY) ** 2)
    opp = math.fabs(leftX - rightX)

    rad = math.asin(opp / hyp)

    # Why am I subtracting 90 here?
    return int(math.degrees(rad) - 90)

def to_bw_pixels(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgPil = Image.fromarray(img)
    return imgPil.convert('1')

def rotate(image, angle):
    (h, w) = image.shape[:2]

    center = (w / 2, h / 2)

    # Perform the rotation
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, mat, (w, h))

    return rotated

# Defining main function
def main():
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--face", help = "The location of your face file")
    args = parser.parse_args()

    face = args.face

    img = cv2.imread(face)

    face, landmarks = find_face_and_features(img)

    adjustment = deg_off_level(landmarks)
    print(adjustment)

    img = rotate(img, adjustment)

    img = to_bw_pixels(img)
    img.show()

    #img = crop_to_face(img, face)

    #get_top_bottom(img, face)

    #top, bottom = get_top_bottom(img, face)

    #top = to_bw_pixels(top)

    #bottom = to_bw_pixels(bottom)

    #top.show() 
    #bottom.show()



# Using the special variable 
# __name__
if __name__=="__main__":
    main()