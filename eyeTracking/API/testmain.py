# test for our library
#import foo.bar

# or
from PupilTracker import PupilTracker

import numpy as np
import cv2
import sys

# test create a class instance
myObject = PupilTracker.TestClass(                                 \
    'C:\opencv\data\haarcascades\haarcascade_frontalface_alt.xml', \
    'C:\opencv\data\haarcascades\haarcascade_eye_tree_eyeglasses.xml')

while(True):
    
    # compute
    diff = myObject.findPupil()
    
    # get different data
    faces = myObject.getFaces()
    eyes  = myObject.getEyes()
    print('Eyes: ', eyes)
    #print('Faces: ', faces)
    
    #print('Main Diff: ', diff)
    
    # get our desired data
    myObject.drawFrame()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break