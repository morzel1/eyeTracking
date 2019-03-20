# required imports
import numpy as np
import cv2
import sys

# test class
# note - python doesn't have private variables, but the
# convention is to name the variable like '_this'
# (preceded by an underscore)
class TestClass:
    
    # constructor
    # args: string of the location of face and eye cascades
    def __init__(self, faceCascade, eyeCascade):
        # load the camera
        self._cam = cv2.VideoCapture(0)
        print('Loaded video capture 0!')
        
        # create cascades
        self._faceCascade = cv2.CascadeClassifier(faceCascade)
        self._eyeCascade  = cv2.CascadeClassifier(eyeCascade)
        
        # create internal variables
        self._SCREEN_WIDTH  = 1920
        self._SCREEN_HEIGHT = 1080
        
        # empty image placeholder
        self._clip = np.zeros((512,512,3), np.uint8)

        # holds the past centers of the eyeball, to use for averaging
        self._centers = []
        
        # holds the rectangle for each detected face
        self._faces = []
        
        # holds the rectangle for each detected eye
        self._eyes = []

        #center = [0,0]

        # mouse speed handling
        #self._mouseAccel = [0,0]
        #self._mouseVel = [0,0]

        # holds the location of the mouse
        self._mousePoint = [self._SCREEN_WIDTH / 2, self._SCREEN_HEIGHT / 2]

        # how far away from the center the pupil has to be in order to move the mouse
        self._MOUSE_THRESH_X = 2
        self._MOUSE_THRESH_Y = 1
    
    # draw the frame with detectors on it
    def drawFrame(self):
        if not self._frame is None and self._frame.any():
            cv2.imshow('frame', self._frame)
    
    # deconstructor
    def __del__(self):
        # close the camera
        self._cam.release()
        cv2.destroyAllWindows()
        print('Destroyed cam and windows!')
    
    # get the face rects
    def getFaces(self):
        return self._faces
        
    # get eye rects
    def getEyes(self):
        return self._eyes
    
    # find the pupil
    def findPupil(self):
    
        # returned
        diff = [0,0]
    
        ret, self._frame = self._cam.read()
        self._frame = cv2.flip(self._frame,1)

        if not ret:
            print 'Error reading frame!'
            sys.exit(-1)

        # convert the frame to grayscale
        gray = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)

        # get the rectangles defining the face
        self._faces = self.detectFaces(gray, self._faceCascade)
        
         # iterate through each detected face
        for face in self._faces:
            # draw a rectangle around each face
            cv2.rectangle(self._frame, (face[0],face[1]), (face[0]+face[2],face[1]+face[3]), (255,0,0), 2)

            # crop the face to only look there for eyes
            fx = face[0]
            fy = face[1]
            fw = face[2]
            fh = face[3]
            #face_cropped = gray[fy:fy + fh, fx:fx + fw]
            face_cropped = gray[fy:fy+fh, fx:fx+fw]

            # detect eyes from the cropped face
            self._eyes = self.detectEyes(face_cropped, self._eyeCascade)

            # draw squares around the eyes
            if not (self._eyes is None):

                # test getting left eye
                self._eyes = np.asarray(self.getLeftEye(self._eyes))
                #print eyes

                if (not (self._eyes is None)) and not len(self._eyes) == 0:
                    eye = self._eyes
                    #for eye in eyes:
                    #for ex, ey, ew, eh in eyes:
                    ex = eye[0]
                    ey = eye[1]
                    ew = eye[2]
                    eh = eye[3]
                    #cv2.rectangle(face_cropped, (ex,ey), (ex+ew,ey+eh),(0,255,0),2)

                    # set the default eye center to be the center of the eye rectangle
                    #defaultEyeCenter[0] = ew / 2
                    #defaultEyeCenter[1] = eh / 2

                    #print('Default eye center: ', defaultEyeCenter )

                    # draw a rectangle around the found eye
                    cv2.rectangle(self._frame, (fx+ex,fy+ey), (fx+ex+ew,fy+ey+eh),(0,255,0),2)

                    # test show for now
                    #cv2.imshow('frame', frame)
                    #return self._frame
                    
                    cex = ex
                    cey = ey + (eh / 4)
                    cew = ew
                    ceh = 2 * eh / 4

                    # clip out just the eye section
                    eye_cropped = face_cropped[ey:ey+eh, ex:ex+ew]
                    #eye_cropped = face_cropped[cey:cey+ceh, cex:cex+cew]

                    # detect circles within the eye crop
                    irises = self.detectIrises(eye_cropped)

                    defaultEyeCenter = [0,0]
                    defaultEyeCenter[0] = fx+cex+(cew/2)
                    defaultEyeCenter[1] = fy+cey+(ceh/2)

                    cv2.circle(self._frame, (defaultEyeCenter[0], defaultEyeCenter[1]), 2, (255, 255, 0), 2)
                    
                    # go through each iris
                    if not (irises is None):
                        irises = np.uint16(np.around(irises))
                        '''for iris in irises[0,:]:
                            ix = iris[0]
                            iy = iris[1]
                            ir = iris[2]

                            #cv2.circle(frame, (int(fx+cex+ix),int(fy+cey+iy)), ir, (0,0,255), 2)
                            #cv2.circle(frame, (fx + cex + ix, fy + cey + iy), ir, (0, 0, 255), 2)
                            cv2.circle(frame, (fx + ex + ix, fy + ey + iy), ir, (0, 0, 255), 2)'''

                        # get the one that is the darkest and call
                        # it the eyeball
                        eyeball = self.getEyeBall(eye_cropped, irises)
                        #print 'eye cropped shape: ', eye_cropped.shape
                        #print eyeball

                        # store the default eye center if it isn't set yet
                        #if len(defaultEyeCenter) == 0 and not (eyeball is None):
                        #    defaultEyeCenter.append(  eyeball[0]  )
                        #    defaultEyeCenter.append(  eyeball[1]  )
                        #    print('default center: ', defaultEyeCenter)

                        # add the current center to the list of centers
                        self._centers.append((eyeball[0], eyeball[1]))

                        # calculate the new average center
                        center = self.stabilize(self._centers, 5) # use the past 5 entries
                        #center = [0,0]
                        #center[0] = eyeball[0]
                        #center[1] = eyeball[1]
                        
                        # calculate the new mouse position
                        if not (center is None):
                            #diff = [0,0]
                            #diff[0] = (center[0] - lastPoint[0]) * MOUSE_SCALE_X
                            #diff[1] = (center[1] - lastPoint[1]) * MOUSE_SCALE_Y
                            #diff[0] = center[0] - defaultEyeCenter[0]
                            #diff[1] = center[1] - defaultEyeCenter[1]
                            diff[0] = (fx+ex+center[0]) - defaultEyeCenter[0]
                            diff[1] = (fy+ey+center[1]) - defaultEyeCenter[1]

                            #if diff[0] > MOUSE_THRESH_X:
                            #    mouseAccel[0] = 2
                            #elif diff[0] < -1*MOUSE_THRESH_X:
                            #    mouseAccel[0] = -2

                            #if ((fy+ey+center[1]) - defaultCenter[1]) > MOUSE_THRESH:
                            #if diff[1] > MOUSE_THRESH_Y:
                            #    mouseAccel[1] = 2
                            #elif diff[1] < -1*MOUSE_THRESH_Y:
                            #    mouseAccel[1] = -2

                            #print('Diff: ', diff)

                            # max difference x
                            #if diff[0] > 5:
                            #    diff[0] = 5
                            #elif diff[1] < -5:
                            #    diff[0] = -5

                            # max difference y
                            #if diff[1] > 5:
                            #    diff[1] = 5
                            #elif diff[1] < -5:
                            #    diff[1] = -5

                            #print('Center: ', center)

                            #print 'Diff: ', diff

                            # move the mouse the difference
                            '''if diff[0] > 1:
                                mousePoint[0] += diff[0] * MOUSE_SCALE_X
                            elif diff[0] < 0:
                                mousePoint[0] += diff[0] * MOUSE_SCALE_X

                            if diff[1] > 1 or diff[1] < -1:
                                mousePoint[1] -= diff[1] * MOUSE_SCALE_Y

                            if( mousePoint[0] > SCREEN_WIDTH ):
                                mousePoint[0] = SCREEN_WIDTH
                            elif( mousePoint[0] < 0):
                                mousePoint[0] = 0
                            if (mousePoint[1] > SCREEN_HEIGHT):
                                mousePoint[1] = SCREEN_HEIGHT
                            elif (mousePoint[1] < 0):
                                mousePoint[1] = 0'''

                            # ratio: center_x / eye_width = mouse_x / screen_width
                            #mousePoint[0] = SCREEN_WIDTH * center[0] / ew
                            # ratio: center_y / eye_height = mouse_y / screen_height
                            #mousePoint[1] = SCREEN_HEIGHT * center[1] / eh
                            #print 'Mousepoint: ', mousePoint

                            #win32api.SetCursorPos((mousePoint[0],mousePoint[1]))

                            lastPoint = center

                        # draw the eyeball circle
                        #if not (eyeball is None):
                            #cv2.circle(frame, (fx + ex + eyeball[0], fy + ey + eyeball[1]), eyeball[2], (0,0,255), 2)
                            #cv2.circle(frame, (fx+ex+center[0],fy+ey+center[1]), eyeball[2], (0,0,255), 2)
                        #    cv2.circle(frame, (fx + ex + center[0], fy + ey + center[1]), 10, (0, 0, 255), 2)
                        cv2.circle(self._frame, (fx + ex + center[0], fy + ey + center[1]), 10, (0, 0, 255), 2)

                        # set the current center as the last point
                        #lastPoint = center

                    # testing eye clipping
                    clip = eye_cropped
                    
        return diff # end findPupil
        
    ############################################
    ##           internal functions           ##
    ############################################
    def getLeftEye(self, eyes):
        i = 0
        leftMost = 99999999
        leftMostIndex = -1

        while True:
            if i > len(eyes) - 1:
                break
            if (eyes[i][0] < leftMost):
                leftMost = eyes[i][0]
                leftMostIndex = i

            i = i + 1
        # print(leftMostIndex)
        if not (eyes is None):
            if len(np.asarray(eyes).shape) != 1:
                return eyes[leftMostIndex]
            else:
                return eyes
        else:
            return None
            
    def getLeftCircle(self, Circles):
        i = 0
        leftMost = 99999999
        leftMostIndex = -1

        while True:
            if i >= len(circles) - 1:
                break
            if (circles[0][0][0] < leftMost):
                leftMost = eyes[0][0][0]
                leftMostIndex = i

            i = i + 1
        # print(leftMostIndex)
        return circles[leftMostIndex]
        
    # function to figure out the location of eyes
    def detectFaces(self, frame, faceCascade):
        # convert the frame and store it in grayscale (if necessary)
        if len(frame.shape) != 2:
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = frame
        # equalize the image (enhance contrast) and store it back into itself
        cv2.equalizeHist(grayscale, grayscale)
        # get a list of faces (in rect form)
        faces = faceCascade.detectMultiScale(grayscale, 1.1, 2)

        return faces
    
    def detectEyes(self, frame, eyeCascade):
        
        # convert the frame and store it in grayscale (if necessary)
        if len(frame.shape) != 2:
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = frame
        # equalize the image (enhance contrast) and store it back into itself
        # cv2.equalizeHist(grayscale, grayscale)
        # get a list of eyes (in rect form)
        #eyes = eyeCascade.detectMultiScale(grayscale, 1.1, 2)
        eyes = eyeCascade.detectMultiScale(grayscale)

        return eyes
        
    def detectIrises(self, frame):
        # convert to grayscale if necessary
        if len(frame.shape) != 2:
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = frame
        # increase contrast
        cv2.equalizeHist(grayscale, grayscale)
        # look for circles
        #print 'eye shape: ', grayscale.shape

        # in tango's code, he uses
        # minDist = eye.cols / 8 --> grayscale.shape[1] / 8
        # threshold = 250
        # minArea = 15
        # minRadius = eye.rows / 8 --> grayscale.shape[0] / 8
        # maxRadius = eye.rows / 3 --> grayscale.shape[0] / 3

        # from stack overflow:
        # param1 - the first method specific parameter. In the
        # case of CV_HOUGH_GRADIENT, it is the higher threshold
        # of the two passed to the Canny() edge detector (the
        # lower one is twice smaller)
        # param2 - second method specific parameter. In the case
        # of CV_HOUGH_GRADIENT, it is the accumulator threshold
        # for the circle centers at the detection stage. the
        # smaller it is, the more false circles may be detected.
        # circles, corresponding to the larger accumulator values,
        # will be returned first

        minDist = grayscale.shape[1] / 8
        minRadius = grayscale.shape[0] / 8
        maxRadius = grayscale.shape[0] / 3

        param1 = 250
        param2 = 15

        irises = cv2.HoughCircles(grayscale, cv2.HOUGH_GRADIENT,
                                  1,minDist,param1=param1,param2=param2,
                                  minRadius=minRadius,maxRadius=maxRadius)

        return irises
        
    # looks for the blackest circle
    def getEyeBall(self, frame, circles):
        # stores the total pixel sum values for each circle
        sums = np.zeros(len(circles))

        #print frame
        #print circles

        # loop through the frame's rows
        for y in range(frame.shape[0]):
            # loop through that row (length of the row):
            for x in range(frame.shape[1]):
                # loop through each circle
                for i in range(len(circles)):
                    # get the center point of the circle
                    center = (int(circles[0][i][0]), int(circles[0][i][1]))
                    radius = int(circles[0][i][2])

                    # checks if the pixel is inside the circle, and
                    # if so adds it to the total circle values
                    if(pow(x - center[0], 2) + pow(y - center[1], 2) < pow(radius, 2)):
                        sums[i] += frame[y][x]

        # figure out the smallest sum
        smallestSum   = 9999999
        smallestIndex = -1
        for i in range(len(circles)):
            if sums[i] < smallestSum:
                smallestIndex = i

        return circles[0][smallestIndex]
        # stores the total pixel sum values for each circle
        '''
        sums = np.zeros(len(circles[0]))
        # print frame
        # print circles
        for z in range (len(circles[0])):
            #print 'ran: ', z , 'times'
            # loop through the frame's rows
            for y in range(frame.shape[0]):
                # loop through that row (length of the row):
                for x in range(frame.shape[1]):
                    # loop through each circle
                    for i in range(len(circles[0])):
                        # get the center point of the circle
                        center = (int(circles[0][z][0]), int(circles[0][z][1]))
                        radius = int(circles[0][z][2])

                        # checks if the pixel is inside the circle, and
                        # if so adds it to the total circle values
                        if (pow(x - center[0], 2) + pow(y - center[1], 2) < pow(radius, 2)):
                            sums[z] += frame[y][x]


            # figure out the smallest sum
            smallestSum = 9999999
            smallestIndex = -1
            for i in range(len(circles[0])):
                #print 'sums', sums[i]
                if sums[i] < smallestSum:
                    smallestIndex = i
                    smallestSum = sums[i]


        #print 'smallest amt', sums[smallestIndex]
        return circles[0][smallestIndex]
        '''
        
    # get the last average X amount of circle locations
    # points is the list of circle points, amount
    # is the number of points to average
    def stabilize(self, points, amount):
        sumX = 0
        sumY = 0
        count = 0
        for i in xrange(max(0, len(points)-amount), len(points)):
            sumX += points[i][0] # x
            sumY += points[i][1] # y
            count += 1
        if count > 0:
            sumX /= count
            sumY /= count

        return (sumX, sumY)