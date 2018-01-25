import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from random import *

# Blue contours are the largest contours within the limit set in the code
# Red contours are the contours with the highest extent ratio
# Green are the contours out of the largest who are the most full/solid
DEBUGGING = False
DEBUG_LARGEST = False
DEBUG_EXTENT = False
DEBUG_SOLIDITY = False
DEBUG_SCORE = True
DEBUG_DISTANCE = False

os.system("v4l2-ctl -d /dev/video1"
          " -c brightness={}".format(175 + randint(-5, 5)) +
          " -c contrast=5"
          " -c saturation=83"
          " -c white_balance_temperature_auto=false"
          " -c sharpness=4500"
          " -c backlight_compensation=0"
          " -c exposure_auto=1"
          " -c exposure_absolute=0"
          " -c pan_absolute=0"
          " -c tilt_absolute=0"
          " -c zoom_absolute=0")


class CScore:

    def __init__(self, contour, size, extent, solidity):
        self.contour = contour
        self.size = size
        self.extent = extent
        self.solidity = solidity

    def getScore(self):
        return self.size + self.extent + self.solidity


class DistAngleVision:

    def __init__(self, imgWidth, imgHeight, fov):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.camIntrinsics = np.asarray([[879.4009463329488, 0.0, 341.3659246478685],
                                         [0.0, 906.4264609420143, 241.09943855444936],
                                         [0.0, 0.0, 1.0]])

        # We assume that fx and fy are the same, so we just average the two values in the intrinsics
        self.focalLength = (self.camIntrinsics[0][0] + self.camIntrinsics[0][1]) / 2
        # In inches. Average 13 and 11 so that we can find the 13 x 13 x 11 cube on all sides within margin of error
        self.cubeWidthReal = 12

        self.distortCoeffs = np.asarray([0.24562790316739747, -4.700752268937957, 0.0031650173316281876, -0.0279002999438822, 17.514821989419733])

        # 3D coordinates of the points we think we can find on the box.
        self.objectPoints = np.asarray([[0, 0, 0],
                                        [0, 0, 1],
                                        [1, 0, 1],
                                        [1, 0, 0]], dtype=np.float32)

        # The vector matrices which are drawn to fit the plane of the face we are finding
        self.axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).reshape(-1, 3)

        self.lowerBound = np.array([10, 133, 60])
        self.upperBound = np.array([180, 255, 255])

        self.degreesPerPix = fov / (np.sqrt(imgWidth ** 2 + imgHeight ** 2))

        # Counter for x axis of scatter graph of DEBUG_DISTANCE function
        self.distances = []
        self.xAxis = []
        self.windowsMoved = False

    def processImg(self, img):
        self.source = img

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lowerb=self.lowerBound, upperb=self.upperBound)
        erode = cv2.erode(mask, kernel=self.createKernel(5))
        dilate = cv2.dilate(erode, kernel=self.createKernel(5))
        close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, self.createKernel(7))
        erode2 = cv2.dilate(close, kernel=self.createKernel(3))
        close2 = cv2.morphologyEx(erode2, cv2.MORPH_CLOSE, self.createKernel(10))

        fstream, contours, hierarchy = cv2.findContours(close2, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            notableContours = self.filterContours(contours, num=10)

            if len(notableContours) > 0:
                scores = []
                for i, cs in enumerate(notableContours):
                    scores.append(cs.getScore())

                sortedScores = sorted(zip(scores, notableContours), key=lambda l: l[0], reverse=True)

                if (sortedScores[0][0] > 2.5):
                    bestCScore = sortedScores[0][1]

                    try:
                        cMoments = cv2.moments(bestCScore.contour)
                        centerPoint = (int((cMoments["m10"] / cMoments['m00'])),
                                       int((cMoments["m01"] / cMoments["m00"])))

                        cv2.circle(img, tuple(centerPoint), 4, (255, 255, 255), -1)

                        minRect = cv2.minAreaRect(bestCScore.contour)
                        box = cv2.boxPoints(minRect)
                        box = np.int0(box)
                        cv2.drawContours(img, [box], 0, (140, 110, 255), 2)

                        distance = self.calcRealDistance(minRect[1][0]) * 2
                        angle = self.calcAngle(centerPoint[0])

                        self.distances.append(distance)
                        self.xAxis.append(len(self.distances))

                        cv2.putText(img, "{}".format(distance), (centerPoint[0], centerPoint[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (268, 52, 67), 2, cv2.LINE_AA)
                        cv2.putText(img, "{}".format(angle), (centerPoint[0], centerPoint[1] + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (125, 52, 150), 2, cv2.LINE_AA)

                    except ZeroDivisionError:
                        pass

                if (DEBUGGING):
                    self.showDebugStatements(notableContours)

        cv2.imshow("Source", self.source)
        cv2.imshow("Color Filtered", close2)

        if (not self.windowsMoved):
            cv2.moveWindow("Source", 75, 0)
            cv2.moveWindow("Color Filtered", 75, 550)
            self.windowsMoved = True

    def createKernel(self, size):
        return np.ones((size, size), np.uint8)

    def getCoordDist(self, p1, p2):
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def getCoordHeight(self, pLeft, pTop, pRight):
        a = self.getCoordDist(pTop, pRight)
        b = self.getCoordDist(pLeft, pRight)
        c = self.getCoordDist(pLeft, pTop)

        theta = np.arccos((c ** 2 - a ** 2 - b ** 2) / (-2 * a * b))

        height = a * np.sin(theta)
        return height

    def drawPnPAxes(self, img, corners, imgPts):
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgPts[0].ravel()), (255, 0, 0), 5)
        img = cv2.line(img, corner, tuple(imgPts[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(imgPts[2].ravel()), (0, 0, 255), 5)
        return img

    # Filters out contours based on 3 parameters:
    #   Size of the contour
    #       Anything with an area less than 1000 pixels is ignored. The {num} largest contours are then put away into a scoring object
    #   Extent of contour
    #   Solidity of contour
    def filterContours(self, contours, num):
        cAreas = []
        scores = []

        for i, c in enumerate(contours):
            cAreas.append(cv2.contourArea(c))

        sortedAreas = sorted(zip(cAreas, contours), key=lambda l: l[0], reverse=True)
        largestArea = sortedAreas[0][0]

        for i in range(len(sortedAreas)):

            if (sortedAreas[i][0] > 1000):
                area = cv2.contourArea(sortedAreas[i][1])

                x, y, w, h = cv2.boundingRect(sortedAreas[i][1])
                rectArea = w * h
                extent = (float(area) / rectArea)

                hull = cv2.convexHull(sortedAreas[i][1])
                hullArea = cv2.contourArea(hull)
                solidity = (float(area) / hullArea)

                relativeArea = sortedAreas[i][0] / largestArea

                cScore = CScore(sortedAreas[i][1], relativeArea, extent, solidity)
                scores.append(cScore)

            if (len(scores) >= num):
                break

        return scores

    def calcRealDistance(self, pxWidth):
        return (self.cubeWidthReal * self.focalLength) / pxWidth

    def calcAngle(self, centerX):
        return self.degreesPerPix * centerX

    def showDebugStatements(self, scores):
        l = 0
        e = 0
        s = 0
        sc = 0

        for score in scores:
            try:
                cMoments = cv2.moments(score.contour)
                centerPoint = (int((cMoments["m10"] / cMoments['m00'])),
                               int((cMoments["m01"] / cMoments["m00"])))
            except ZeroDivisionError:
                pass

            if (DEBUG_LARGEST):
                cv2.drawContours(self.source, score.contour, -1, (255, 0, 0), 2)
                cv2.putText(self.source, "{}".format(l), tuple(centerPoint), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(self.source, "{}".format(score.size), (centerPoint[0] + 25, centerPoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                l += 1

            if (DEBUG_EXTENT):
                cv2.drawContours(self.source, score.contour, -1, (0, 0, 255), 2)
                cv2.putText(self.source, "{}".format(e), (centerPoint[0], centerPoint[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(self.source, "{}".format(score.extent), (centerPoint[0] + 25, centerPoint[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                e += 1

            if (DEBUG_SOLIDITY):
                cv2.drawContours(self.source, score.contour, -1, (0, 255, 0), 2)
                cv2.putText(self.source, "{}".format(s), (centerPoint[0], centerPoint[1] - 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(self.source, "{}".format(score.solidity), (centerPoint[0] + 25, centerPoint[1] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                s += 1

            if (DEBUG_SCORE):
                cv2.drawContours(self.source, score.contour, -1, (255, 255, 255), 2)
                cv2.putText(self.source, "{}".format(sc), (centerPoint[0], centerPoint[1] - 105), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(self.source, "{}".format(score.getScore()), (centerPoint[0] + 25, centerPoint[1] - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                sc += 1

            if (DEBUG_DISTANCE):
                plt.scatter(self.xAxis, self.distances)
                plt.pause(float(1) / 30)

    def getSourceImg(self):
        return self.source


if __name__ == '__main__':

    for x in range(1, 5):
        stream = cv2.VideoCapture(x)

        if (stream.isOpened()):
            print("Camera found on port: %d" % (x))
            break;

    if (not stream.isOpened()):
        print("Camera not found")
        sys.exit()

    # Lifecam is 60 degrees from left to right. Pass it only half of fov
    vision = DistAngleVision(stream.get(cv2.CAP_PROP_FRAME_WIDTH), stream.get(cv2.CAP_PROP_FRAME_HEIGHT), 30)

    while True:
        ret, src = stream.read()

        plt.ion()
        axes = plt.gca()
        axes.set_ylim([0, 500])

        if (ret):

            vision.processImg(src)

            keyPressed = cv2.waitKey(33)
            if (keyPressed == ord("s")):
                cv2.imwrite("{}.png".format("PnPVisionImg"), vision.getSourceImg())
            elif (keyPressed == ord("q")):
                cv2.destroyAllWindows()
                sys.exit()
