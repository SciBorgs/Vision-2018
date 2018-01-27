import numpy as np
import cv2
import sys
import os
from enum import Enum
from random import *

os.system("v4l2-ctl -d /dev/video1"
          " -c brightness={}".format(10 + randint(-5, 5)) +
          " -c white_balance_temperature_auto=false"
          " -c exposure_auto=1")

class Direction(Enum):
    RIGHT = 0
    LEFT = 1

class PnPVision:


    def __init__(self, lbound, ubound):
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.camIntrinsics = np.asarray([[879.4009463329488, 0.0, 341.3659246478685],
                                    [0.0, 906.4264609420143, 241.09943855444936],
                                    [0.0, 0.0, 1.0]])

        self.distortCoeffs = np.asarray([0.24562790316739747, -4.700752268937957, 0.0031650173316281876, -0.0279002999438822, 17.514821989419733])

        # 3D coordinates of the points we think we can find on the box.

        self.objectPoints = np.asarray([[-0.5416666667, 0, -0.5416666667],
                                   [-0.5416666667, 0, 0.5416666667],
                                   [0.5416666667, 0, 0.5416666667],
                                   [0.5416666667, 0, -0.5416666667]], dtype=np.float32)

        self.objectPoints = np.asarray([[0, 0, 0],
                                   [0, 0, 1.08333333],
                                   [1.08333333, 0, 1.08333333],
                                   [1.08333333, 0, 0]], dtype=np.float32)

        # The vector matrices which are drawn to fit the plane of the face we are finding
        self.axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).reshape(-1, 3)

        self.headOnThresh = 200
        self.slightlyHeadonThresh = 50
        self.lowerBound = lbound
        self.upperBound = ubound





    #frame_counter = 0
    def processImg(self, img):

        self.source = img

        #Filtering
        hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lowerBound, upperBound)
        erode = cv2.erode(mask, create_kernel(5))
        dilate = cv2.dilate(erode, create_kernel(5))
        closed = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, create_kernel(7))
        erode2 = cv2.dilate(closed, create_kernel(3))
        close2 = cv2.morphologyEx(erode2, cv2.MORPH_CLOSE, create_kernel(10))

        # Combine original image and filtered mask
        compound = cv2.bitwise_and(source, source, mask=closed)

        #Contours
        fstream, contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        largestAreaIndex = 0
        i = 0

        if len(contours) > 0:
                
            cAreas = []
        
            for i, c in enumerate(contours):   
                cAreas.append(cv2.contourArea(c))
        
            sortedAreas = sorted(zip(cAreas, contours), key=lambda l: l[0], reverse=True)
            largestArea = sortedAreas[0][0]
            contour = sortedAreas[0][1]

            #Approx Poly dp
            epsilon = 0.03 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            img = connect(img, approx)
            for i, approxPoint in enumerate(approx):
                img = cv2.circle(img, tuple(approxPoint[0]), 4, (0, 0, 255), -1)
                text(img, str(i + 1), tuple(approxPoint[0]), (0, 0, 0))
                pass

            #Assigning points initially - they stay these values if the cube is not head on or slightly head on
            topPoint = tuple(contour[contour[:, :, 1].argmin()][0])
            rightPoint = tuple(contour[contour[:, :, 0].argmax()][0])
            leftPoint = tuple(contour[contour[:, :, 0].argmin()][0])
            botPoint = tuple(contour[contour[:, :, 1].argmax()][0])

            #Finding the midPoint (the point of the corner of the box not on the edge of the contour)
            midHeight = findHeight(leftPoint, topPoint, rightPoint)
            midPoint = tuple([botPoint[0], topPoint[1] + 2 * midHeight])

            #Find if the cube is head on or slightly head on
            facingHeadon = False
            slightlyHeadon = False
            slightlyHeadonDirection = Direction.LEFT

            if len(approx) <= 4 and len(approx) > 1:
                sortedArr = sortByColumn(approx, column=1, flipped=False)
                if abs(sortedArr[0].ravel()[1] - sortedArr[1].ravel()[1]) < headOnThresh:
                    facingHeadon = True
            elif len(approx) == 5:
                sortedArr = sortByColumn(approx, column=1, flipped=False)
                if abs(sortedArr[0].ravel()[1] - sortedArr[1].ravel()[1]) < slightlyHeadonThresh :
                    slightlyHeadon = True

            #Headon (apply directly to the forehead)
            if facingHeadon: 
                sortedTop = sortByColumn(approx, column=1, flipped=False)[:2]
                sortedLeftTop = sortByColumn(sortedTop, column=0)

                midPoint = leftPoint
                leftPoint = tuple(sortedLeftTop[0].ravel())
                topPoint = tuple(sortedLeftTop[1].ravel())

            #Slightly headon
            elif slightlyHeadon:
                sortedTop = sortByColumn(approx, column=1, flipped=False)
                print(sortedTop)
                sortedLeftTop = sortByColumn(sortedTop[:2], column=0)
                mid = sortedTop[2].ravel()
                img = cv2.circle(img, tuple(sortedLeftTop[0].ravel()), 4, (223, 66, 224), -1)
                img = cv2.circle(img, tuple(sortedLeftTop[1].ravel()), 4, (223, 66, 224), -1)

                if abs(mid[0] - sortedLeftTop[0].ravel()[0]) < slightlyHeadonThresh:
                    slightlyHeadonDirection = Direction.LEFT
                    midPoint = leftPoint
                    leftPoint = tuple(sortedLeftTop[0].ravel())
                    topPoint = tuple(sortedLeftTop[1].ravel())
                    rightPoint = tuple([midPoint[0], topPoint[1]])
                elif abs(mid[0] - sortedLeftTop[1].ravel()[0]) < slightlyHeadonThresh:
                    slightlyHeadonDirection = Direction.RIGHT
                    leftPoint = tuple(sortedLeftTop[0].ravel())
                    topPoint = tuple(sortedLeftTop[1].ravel())            
                    midPoint = tuple([leftPoint[0], rightPoint[1]])

            img = cv2.circle(img, topPoint, 4, (0, 0, 255), -1)
            text(img, "topPoint", tuple(topPoint), (0,0,0))
            img = cv2.circle(img, midPoint, 4, (0, 255, 0), -1)
            text(img, "midPoint", tuple(midPoint), (0,0,0))
            img = cv2.circle(img, leftPoint, 4, (0, 0, 0), -1)
            text(img, "leftPoint", tuple(leftPoint), (0,0,0))
            img = cv2.circle(img, rightPoint, 4, (255, 255, 255), -1)
            text(img, "rightPoint", tuple(rightPoint), (0,0,0))



            
            img = cv2.circle(img, midPoint, 4, (0, 0, 255), -1)
            points = np.asarray([leftPoint, topPoint, rightPoint, midPoint], dtype=np.float32)

            points = np.reshape(points, (4, 2))
            objectPoints = np.reshape(objectPoints, (4, 3))
            ret2, rvecs, tvecs = cv2.solvePnP(objectPoints, points, camIntrinsics, distortCoeffs)
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, camIntrinsics, distortCoeffs)
            rmat = None
            rmat, jac2 = cv2.Rodrigues(rvecs, rmat)
            #print(rmat.shape)
            cmat = None

            projMat = np.array(
                [[rmat[0][0], rmat[0][1], rmat[0][2], 0],
                 [rmat[1][0], rmat[1][1], rmat[1][2], 0],
                 [rmat[2][0], rmat[2][1], rmat[2][2], 0]
                ])
            tvecs1 = None
            rmat1 = None
            rotmatX = None
            rotmatY = None
            rotmatZ = None
            eulerAngles = None
            cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles = cv2.decomposeProjectionMatrix(projMat, cmat, rmat1, tvecs1, rotmatX, rotmatY, rotmatZ, eulerAngles)
            #print(eulerAngles)
            distance = np.sqrt(tvecs[0].ravel() ** 2 + tvecs[1].ravel() ** 2 + tvecs[2].ravel() ** 2)
            text(img, "distance: %.2f ft." % (distance), (100, 100), (0, 0, 0))

            sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
            eulerY = np.arctan2(-rmat[2, 0], sy)
            #print((eulerY * (180/np.pi)) % 90)
            res = cv2.bitwise_and(img,img, mask =close2)

            source = drawPnPAxes(source, points, imgpts)
        cv2.imshow("frame2", source)
        cv2.imshow("compound", res)

    def create_kernel(self, size):
        return np.ones((size, size), np.uint8)


    def drawPnPAxes(self, img, corners, imgpts):
        try:
            corner = tuple(corners[0].ravel())
            img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
            img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
            img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
        except Exception as e:
            pass
        return img


    def getCoordDist(self, point1, point2):
        return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


    def findHeight(self, left, top, right):
        a = getCoordDist(top, right)
        b = getCoordDist(left, right)
        c = getCoordDist(left, top)

        theta = np.arccos((c ** 2 - a ** 2 - b ** 2) / (-2 * a * b))

        height = a * np.sin(theta)
        return int(height)

    def connect(self, img, pts):
        for i, pt in enumerate(pts):
            if i < len(pts) - 1:
                img = cv2.line(img, tuple(pt.ravel()), tuple(pts[i + 1].ravel()), (0, 0, 0), 5)
            else:
                img = cv2.line(img, tuple(pt.ravel()), tuple(pts[0].ravel()), (0, 0, 0), 5)
        return img

    def text(self, img, string, pt, clr):
        img = cv2.putText(img, string, pt, cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, clr, 1)
        return img

    def sortByColumn(self, arr, column=0, flipped=False):
        if flipped:
            return arr[np.flip(np.argsort(arr[:,0,column]),0)]
        else:
            return arr[np.argsort(arr[:,0,column])]
        
    def rotate90(self, frame):
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame, 1)
        return frame



if __name__ == "__main__":
    for x in range(1, 5):
        stream = cv2.VideoCapture(x)
    if (len(sys.argv) > 1):
        if sys.argv[1] == '-m':
        if (stream.isOpened()):
            print("Camera found on port: %d" % (x))
            break;

    if (not stream.isOpened()):
        print("Camera not found")
        sys.exit()
    lowerBound = np.array([18, 0, 194])
    upperBound = np.array([42, 255, 255])
    pnpVision = PnPVision()

    while True:
        ret, src = stream.read()

        if ret:
            pnpVision.processImg(src)

            keyPressed = cv2.waitKey(33)
            if (keyPressed == ord("s")):
                cv2.imwrite("{}.png".format("PnPVisionImg"), vision.getSourceImg())
            elif (keyPressed == ord("q")):
                cv2.destroyAllWindows()
                sys.exit()
