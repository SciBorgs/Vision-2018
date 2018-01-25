import numpy as np
import cv2
import sys
import os
from random import *

os.system("v4l2-ctl -d /dev/video1"
          " -c brightness={}".format(175 + randint(-5, 5)) +
          " -c contrast=5"
          " -c saturation=83"
          " -c white_balance_temperature_auto=false"
          " -c sharpness=4500"
          " -c backlight_compensation=0"
          " -c exposure_auto=1"
          " -c exposure_absolute=10"
          " -c pan_absolute=0"
          " -c tilt_absolute=0"
          " -c zoom_absolute=0")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

camIntrinsics = np.asarray([[879.4009463329488, 0.0, 341.3659246478685],
                            [0.0, 906.4264609420143, 241.09943855444936],
                            [0.0, 0.0, 1.0]])

distortCoeffs = np.asarray([0.24562790316739747, -4.700752268937957, 0.0031650173316281876, -0.0279002999438822, 17.514821989419733])

# 3D coordinates of the points we think we can find on the box.
objectPoints = np.asarray([[0, 0, 0],
                           [0, 0, 1.08333333],
                           [1.08333333, 0, 1.08333333],
                           [1.08333333, 0, 0]], dtype=np.float32)

# The vector matrices which are drawn to fit the plane of the face we are finding
axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).reshape(-1, 3)

headOnThresh = 50


def create_kernel(size):
    return np.ones((size, size), np.uint8)


def drawPnPAxes(img, corners, imgpts):
    try:
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    except Exception as e:
        # raise

        pass
    else:
        pass
    finally:
        pass
    return img


def getCoordDist(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def findHeight(left, top, right):
    a = getCoordDist(top, right)
    b = getCoordDist(left, right)
    c = getCoordDist(left, top)

    theta = np.arccos((c ** 2 - a ** 2 - b ** 2) / (-2 * a * b))

    height = a * np.sin(theta)
    return height


def connect(img, pts):
    for i, pt in enumerate(pts):
        if i < len(pts) - 1:
            img = cv2.line(img, tuple(pt.ravel()), tuple(pts[i + 1].ravel()), (255, 0, 0), 5)
        else:
            img = cv2.line(img, tuple(pt.ravel()), tuple(pts[0].ravel()), (255, 0, 0), 5)
    return img


def text(img, string, pt, clr):
    img = cv2.putText(img, string, pt, cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, clr, 1)
    return img


def sortByColumn(arr, column=0, flipped=False):
    if flipped:
        return arr[np.flip(np.argsort(arr[:, 0, column]), 0)]
    else:
        return arr[np.argsort(arr[:, 0, column])]


cap = cv2.VideoCapture(1)
lowerBound = np.array([10, 133, 60])
upperBound = np.array([180, 255, 255])

if (len(sys.argv) > 1):
    if sys.argv[1] == '-m':
        lower_bound = np.array([26, 17, 231])
        upper_bound = np.array([52, 101, 255])

while (True):
    ret, img = cap.read()
    source = img

    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerBound, upperBound)

    erode = cv2.erode(mask, create_kernel(5))
    dilate = cv2.dilate(erode, create_kernel(5))
    closed = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, create_kernel(7))
    erode2 = cv2.dilate(closed, create_kernel(3))
    close2 = cv2.morphologyEx(erode2, cv2.MORPH_CLOSE, create_kernel(10))

    # Combine original image and filtered mask
    compound = cv2.bitwise_and(source, source, mask=closed)

    fstream, contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest contour
    largest_area_index = 0
    i = 0

    if len(contours) == 0:
        continue
    for contour in contours:
        if cv2.contourArea(contour) >= cv2.contourArea(contours[largest_area_index]):
            largest_area_index = i
        i += 1
    contour = contours[largest_area_index]

    minRect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(minRect)
    box = np.int0(box)
    # cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    print(len(approx))
    img = connect(img, approx)
    for i, approxPoint in enumerate(approx):
        # img = cv2.circle(img, tuple(approxPoint[0]), 4, (0, 0, 255), -1)
        text(img, str(i + 1), tuple(approxPoint[0]), (0, 0, 0))

    topPoint = tuple(contour[contour[:, :, 1].argmin()][0])
    rightPoint = tuple(contour[contour[:, :, 0].argmax()][0])
    leftPoint = tuple(contour[contour[:, :, 0].argmin()][0])
    botPoint = tuple(contour[contour[:, :, 1].argmax()][0])

    midHeight = findHeight(leftPoint, topPoint, rightPoint)
    midHeight = int(midHeight)
    # print(midHeight)
    midPoint = tuple([botPoint[0], topPoint[1] + 2 * midHeight])

    facingHeadon = False

    if len(approx) < 6 and len(approx) > 1:
        # for i, pt1 in enumerate(approx):
        #     point1 = pt1.ravel()
        #     for j, pt2 in enumerate(approx):
        #         point2 = pt2.ravel()
        #         if abs(point2[1] - point1[1]) < headOnThresh:
        #             facingHeadon = True
        sortedArr = sortByColumn(approx, column=1, flipped=False)
        if abs(sortedArr[0].ravel()[1] - sortedArr[1].ravel()[1]) < headOnThresh:
            facingHeadon = True

    if facingHeadon:
        sortedTop = sortByColumn(approx, column=1, flipped=False)[:2]
        sortedLeftTop = sortByColumn(sortedTop, column=0)

        midPoint = leftPoint
        leftPoint = tuple(sortedLeftTop[0].ravel())
        topPoint = tuple(sortedLeftTop[1].ravel())
        img = cv2.circle(img, topPoint, 4, (0, 0, 255), -1)
        img = cv2.circle(img, midPoint, 4, (0, 255, 0), -1)
        img = cv2.circle(img, leftPoint, 4, (0, 0, 0), -1)
        img = cv2.circle(img, rightPoint, 4, (255, 255, 255), -1)

    # img = cv2.circle(img, midPoint, 4, (0, 0, 255), -1)
    points = np.asarray([leftPoint, topPoint, rightPoint, midPoint], dtype=np.float32)

    points = np.reshape(points, (4, 2))
    objectPoints = np.reshape(objectPoints, (4, 3))
    ret2, rvecs, tvecs = cv2.solvePnP(objectPoints, points, camIntrinsics, distortCoeffs)
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, camIntrinsics, distortCoeffs)
    rmat = None
    rmat, jac2 = cv2.Rodrigues(rvecs, rmat)

    # print(tvecs)
    distance = np.sqrt(tvecs[0].ravel() ** 2 + tvecs[1].ravel() ** 2 + tvecs[2].ravel() ** 2)
    text(img, "distance: %.2f ft." % (distance), (100, 100), (0, 0, 0))

    sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    eulerY = np.arctan2(-rmat[2, 0], sy)
    # print(eulerY)

    source = drawPnPAxes(source, points, imgpts)
    cv2.imshow("frame2", source)
    cv2.imshow("compound", compound)
    k = cv2.waitKey(5)
    if k == 27:
        break
cv2.destroyAllWindows()
