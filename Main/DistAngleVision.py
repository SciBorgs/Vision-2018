import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from random import randint

# Blue contours are the largest contours within the limit set in the code
# Red contours are the contours with the highest extent ratio
# Green are the contours out of the largest who are the most full/solid
DEBUGGING = True
DEBUG_LARGEST = False
DEBUG_EXTENT = False
DEBUG_SOLIDITY = False
DEBUG_SCORE = True
DEBUG_WH = True
DEBUG_DISTANCE = False

DEBUG_BOX = False

SHOW_OUTPUTS = False

# os.system("v4l2-ctl -d /dev/video1"
#           " -c brightness={}".format(80 + randint(-5, 5)) +
#           " -c white_balance_temperature_auto=false"
#           " -c exposure_auto=1"
#           " -c exposure_absolute=20")


class CScore:

    def __init__(self, contour, size, extent, solidity, whRatio):
        self.contour = contour
        self.size = size
        self.extent = extent
        self.solidity = solidity
        self.whRatio = whRatio

    def getScore(self):
        return self.size + self.extent + self.solidity


class Vision:

    def __init__(self, imgWidth, imgHeight, fov, ntHandler):

        self.camIntrinsics = np.asarray([[879.4009463329488, 0.0, 341.3659246478685],
                                         [0.0, 906.4264609420143, 241.09943855444936],
                                         [0.0, 0.0, 1.0]])

        # We assume that fx and fy are the same, so we just average the two values in the intrinsics
        self.focalLength = (self.camIntrinsics[0][0] + self.camIntrinsics[0][1])
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

        # Class room
        # self.lowerBound = np.array([17, 163, 70])
        # self.upperBound = np.array([30, 219, 227])

        # Staircase
        # self.lowerBound = np.array([0, 186, 64])
        # self.upperBound = np.array([180, 255, 255])

        #LA Example
        self.lowerBound = np.array([98, 38, 94])
        self.upperBound = np.array([136, 124, 184])

        self.resolution = {"width": imgWidth, "height": imgHeight}
        self.degreesPerPix = fov / (np.sqrt(imgWidth ** 2 + imgHeight ** 2))

        self.ntHandler = ntHandler

        # Counter for x axis of scatter graph of DEBUG_DISTANCE function
        self.allDistances = []
        self.xAxis = []
        self.windowsMoved = False

        self.allBoxes = []

    def processImg(self, img):
        self.source = img

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lowerb=self.lowerBound, upperb=self.upperBound)

        erode = cv2.erode(mask, kernel=self.createKernel(3))
        dilate = cv2.dilate(erode, kernel=self.createKernel(3))
        close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, self.createKernel(15))
        erode2 = cv2.dilate(close, kernel=self.createKernel(7))

        fstream, contours, hierarchy = cv2.findContours(erode2, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        canSeeCube = False
        if len(contours) > 0:
            notableContours = self.filterContours(contours, num=10)

            if len(notableContours) > 0:
                scores = []
                for i, cs in enumerate(notableContours):
                    scores.append(cs.getScore())

                # List of [score, CScore] lists
                sortedScores = sorted(zip(scores, notableContours), key=lambda l: l[0], reverse=True)

                for n in range(len(sortedScores)):

                    if (sortedScores[n][0] > 2.5):

                        if (0.7 < sortedScores[n][1].whRatio < 1.3):
                            try:
                                canSeeCube = True
                                cMoments = cv2.moments(sortedScores[n][1].contour)
                                centerPoint = (int((cMoments["m10"] / cMoments['m00'])),
                                               int((cMoments["m01"] / cMoments["m00"])))

                                cv2.circle(img, tuple(centerPoint), 4, (255, 255, 255), -1)

                                minRect = cv2.minAreaRect(sortedScores[n][1].contour)
                                box = cv2.boxPoints(minRect)
                                box = np.int0(box)
                                cv2.drawContours(img, [box], 0, (140, 110, 255), 2)

                                distance = self.calcRealDistance(minRect[1][0])
                                angle = self.calcAngle(centerPoint[0])
                                self.ntHandler.setValue("distanceToCube", distance)
                                self.ntHandler.setValue("angleToCube", angle)

                                cv2.putText(img, "Power Cube", (centerPoint[0], centerPoint[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

                                if (SHOW_OUTPUTS):
                                    cv2.putText(img, "{0:.2f}".format(distance), (centerPoint[0], centerPoint[1] + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (268, 52, 67), 2, cv2.LINE_AA)
                                    cv2.putText(img, "{0:.2f}".format(angle), (centerPoint[0], centerPoint[1] + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)

                                if (DEBUGGING):
                                    if (DEBUG_DISTANCE):
                                        self.allDistances.append(distance)
                                        self.xAxis.append(len(self.allDistances))

                                    if (DEBUG_BOX):
                                        cv2.circle(self.source, tuple(box[0]), 4, (255, 0, 0), -1)
                                        cv2.circle(self.source, tuple(box[1]), 4, (0, 255, 0), -1)
                                        cv2.circle(self.source, tuple(box[2]), 4, (0, 0, 255), -1)
                                        cv2.circle(self.source, tuple(box[3]), 4, (0, 0, 0), -1)

                            except ZeroDivisionError:
                                pass

                if (DEBUGGING):
                    self.showDebugStatements(notableContours)

        cv2.imshow("Source", cv2.resize(self.source, (0, 0), fx=1.5, fy=1.5))
        cv2.imshow("Color Filtered", cv2.resize(erode2, (0, 0), fx=1.5, fy=1.5))

        self.ntHandler.setValue("canSeeCube", canSeeCube)

        if (not self.windowsMoved):
            cv2.moveWindow("Source", 75, 0)
            cv2.moveWindow("Color Filtered", 1000, 550)
            self.windowsMoved = True

    def createKernel(self, size):
        return np.ones((size, size), np.uint8)

    def filterContours(self, contours, num):
        cAreas = []
        scores = []

        for i, c in enumerate(contours):
            cAreas.append(cv2.contourArea(c))

        # sortedAreas[i] designates separate lists of [area, contour]
        sortedAreas = sorted(zip(cAreas, contours), key=lambda l: l[0], reverse=True)
        largestArea = sortedAreas[0][0]

        for i in range(len(sortedAreas)):

            if (sortedAreas[i][0] > 1000):
                area = sortedAreas[i][0]

                x, y, w, h = cv2.boundingRect(sortedAreas[i][1])
                rectArea = w * h
                extent = (float(area) / rectArea)

                hull = cv2.convexHull(sortedAreas[i][1])
                hullArea = cv2.contourArea(hull)
                solidity = (float(area) / hullArea)

                relativeArea = (float(area) / largestArea)

                minRect = cv2.minAreaRect(sortedAreas[i][1])
                box = cv2.boxPoints(minRect)
                box = np.int0(box)
                minWHRatio = np.sqrt(((box[0][1] - box[1][1]) ** 2 + (box[0][0] - box[1][0]) ** 2) /
                                     ((box[0][1] - box[3][1]) ** 2 + (box[0][0] - box[3][0]) ** 2))

                cScore = CScore(sortedAreas[i][1], relativeArea, extent, solidity, minWHRatio)
                scores.append(cScore)

            if (len(scores) >= num):
                break

        return scores

    def calcRealDistance(self, pxWidth):
        return (self.cubeWidthReal * self.focalLength) / pxWidth

    def calcAngle(self, centerX):
        centerX -= self.resolution["width"] / 2
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
                cv2.putText(self.source, "{0:.2f}".format(score.size), (centerPoint[0] + 25, centerPoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                l += 1

            if (DEBUG_EXTENT):
                cv2.drawContours(self.source, score.contour, -1, (0, 0, 255), 2)
                cv2.putText(self.source, "{}".format(e), (centerPoint[0], centerPoint[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(self.source, "{0:.2f}".format(score.extent), (centerPoint[0] + 25, centerPoint[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                e += 1

            if (DEBUG_SOLIDITY):
                cv2.drawContours(self.source, score.contour, -1, (0, 255, 0), 2)
                cv2.putText(self.source, "{}".format(s), (centerPoint[0], centerPoint[1] - 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(self.source, "{0:.2f}".format(score.solidity), (centerPoint[0] + 25, centerPoint[1] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                s += 1

            if (DEBUG_SCORE):
                cv2.drawContours(self.source, score.contour, -1, (255, 255, 255), 2)
                cv2.putText(self.source, "{}".format(sc), (centerPoint[0], centerPoint[1] - 105), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(self.source, "{0:.2f}".format(score.getScore()), (centerPoint[0] + 25, centerPoint[1] - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                sc += 1

            if (DEBUG_WH):
                cv2.putText(self.source, "{0:.2f}".format(score.whRatio), (centerPoint[0] + 50, centerPoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 125, 125), 2, cv2.LINE_AA)

            if (DEBUG_DISTANCE):
                plt.scatter(self.xAxis, self.allDistances)
                plt.pause(float(1) / 30)

    def getSourceImg(self):
        return self.source
