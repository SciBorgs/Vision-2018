import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Blue contours are the largest contours within the limit set in the code
# Red contours are the contours with the highest extent ratio
# Green are the contours out of the largest who are the most full/solid
DEBUG_LARGEST = True
DEBUG_EXTENT = True
DEBUG_SOLID = True
DEBUG_MAIN = True
DEBUG_DISTANCE = False

class PnPVision:

    def __init__(self):
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

        # os.system("v4l2-ctl -d /dev/video0
        # -c exposure_auto=1
        # -c white_balance_temperature_auto=0
        # -c brightness=0
        # -c exposure_absolute=20")
        self.lowerBound = np.array([10, 133, 60])
        self.upperBound = np.array([180, 255, 255])

        self.n = 0

    def processImg(self, img):
        self.source = img
        xAxis = []
        yAxis = []

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lowerb=self.lowerBound, upperb=self.upperBound)
        erode = cv2.erode(mask, kernel=self.createKernel(5))
        dilate = cv2.dilate(erode, kernel=self.createKernel(5))
        close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, self.createKernel(7))

        fstream, contours, hierarchy = cv2.findContours(close, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            largest = self.filterSize(contours, num=5)
            highestExtent = self.filterExtent(largest, minPercent=50)
            mostSolid = self.filterFullness(highestExtent, minPercent=80)

            for c in mostSolid:
                try:
                    cMoments = cv2.moments(c)
                    centerPoint = (int((cMoments["m10"] / cMoments['m00'])),
                                   int((cMoments["m01"] / cMoments["m00"])))
                    cv2.circle(img, tuple(centerPoint), 4, (255, 255, 255), -1)

                    minRect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(minRect)
                    box = np.int0(box)
                    cv2.drawContours(img, [box], 0, (140, 110, 255), 2)

                    distance = self.calcRealDistance(minRect[1][0]) * 2

                    if (DEBUG_MAIN):
                        # Light green is adjacent to pink and red
                        # Light Green
                        cv2.circle(img, tuple(box[0]), 4, (3, 188, 76), -1)
                        # Red
                        cv2.circle(img, tuple(box[1]), 4, (8, 22, 240), -1)
                        # Yellow
                        cv2.circle(img, tuple(box[2]), 4, (92, 240, 255), -1)
                        # Pink
                        cv2.circle(img, tuple(box[3]), 4, (137, 112, 255), -1)

                    if (DEBUG_DISTANCE):
                        xAxis.append(self.n)
                        yAxis.append(distance)

                        axes = plt.gca()
                        axes.set_ylim([0, 500])

                        plt.scatter(xAxis, yAxis)

                        plt.pause(0.01)
                        self.n += 1

                    cv2.putText(img, "{}".format(distance), (centerPoint[0], centerPoint[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (268, 52, 67), 2, cv2.LINE_AA)

                except ZeroDivisionError:
                    pass

        cv2.imshow("Source", img)
        cv2.imshow("Color Filtered", close)
        cv2.moveWindow("Color Filtered", 75, 550)

        self.source = img

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

    # Returns the largest {num} contours on the screen
    def filterSize(self, contours, num):
        cAreas = []
        largest = []
        n = 0

        for i, c in enumerate(contours):
            cAreas.append(cv2.contourArea(c))

        sortedAreas = sorted(zip(cAreas, contours), key=lambda l: l[0], reverse=True)

        for i in range(len(sortedAreas)):

            if (sortedAreas[i][0] > 1000):
                largest.append(sortedAreas[i][1])

            if (len(largest) >= num):
                break

        if (DEBUG_LARGEST):
            largestImg = self.source
            for c in largest:
                try:
                    cMoments = cv2.moments(c)
                    centerPoint = (int((cMoments["m10"] / cMoments['m00'])),
                                   int((cMoments["m01"] / cMoments["m00"])))
                except ZeroDivisionError:
                    pass

                cv2.drawContours(largestImg, c, -1, (255, 0, 0), 2)
                cv2.putText(largestImg, "{}".format(n), tuple(centerPoint), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                n += 1

        return largest

    def filterExtent(self, contours, minPercent):
        mostRect = []
        n = 0

        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            rectArea = w * h
            extent = (float(area) / rectArea) * 100

            if extent >= minPercent:
                mostRect.append(c)

            if (DEBUG_EXTENT):
                extentImg = self.source

                try:
                    cMoments = cv2.moments(c)
                    centerPoint = (int((cMoments["m10"] / cMoments['m00'])),
                                   int((cMoments["m01"] / cMoments["m00"])))
                except ZeroDivisionError:
                    pass

                cv2.drawContours(extentImg, c, -1, (0, 0, 255), 2)
                cv2.putText(extentImg, "{}".format(n), (centerPoint[0], centerPoint[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(extentImg, "{}".format(int(extent)), (centerPoint[0] + 25, centerPoint[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

                n += 1

        return mostRect

    def filterFullness(self, contours, minPercent):
        solidContours = []
        n = 0

        for c in contours:
            cArea = cv2.contourArea(c)
            hull = cv2.convexHull(c)
            hullArea = cv2.contourArea(hull)

            solidity = (float(cArea) / hullArea) * 100

            if solidity >= minPercent:
                solidContours.append(c)

            if (DEBUG_SOLID):
                fullnessImg = self.source

                try:
                    cMoments = cv2.moments(c)
                    centerPoint = (int((cMoments["m10"] / cMoments['m00'])),
                                   int((cMoments["m01"] / cMoments["m00"])))
                except ZeroDivisionError:
                    pass

                cv2.drawContours(fullnessImg, c, -1, (0, 255, 0), 2)
                cv2.putText(fullnessImg, "{}".format(n), (centerPoint[0], centerPoint[1] - 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(fullnessImg, "{}".format(int(solidity)), (centerPoint[0] + 25, centerPoint[1] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

                n += 1

        return solidContours

    def calcRealDistance(self, pxWidth):
        return (self.cubeWidthReal * self.focalLength) / pxWidth

    def getSourceImg(self):
        return self.source


if __name__ == '__main__':

    for x in range(0, 5):
        stream = cv2.VideoCapture(x)

        if (stream.isOpened()):
            print("Camera found on port: %d" % (x))
            break;

    if (not stream.isOpened()):
        print("Camera not found")
        sys.exit()

    vision = PnPVision()

    while True:
        ret, src = stream.read()

        plt.ion()

        if (ret):

            vision.processImg(src)

            keyPressed = cv2.waitKey(33)
            if (keyPressed == ord("s")):
                cv2.imwrite("{}.png".format("PnPVisionImg"), vision.getSourceImg())
            elif (keyPressed == ord("q")):
                cv2.destroyAllWindows()
                sys.exit()
