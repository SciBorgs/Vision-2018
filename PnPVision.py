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

class Direction(Enum):
    RIGHT = 0
    LEFT = 1

headOnThresh = 200
slightlyHeadonThresh = 50

def create_kernel(size):
    return np.ones((size, size), np.uint8)


def drawPnPAxes(img, corners, imgpts):
    try:
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    except Exception as e:
        #raise

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
            img = cv2.line(img, tuple(pt.ravel()), tuple(pts[i + 1].ravel()), (0, 0, 0), 5)
        else:
            img = cv2.line(img, tuple(pt.ravel()), tuple(pts[0].ravel()), (0, 0, 0), 5)
    return img

def text(img, string, pt, clr):
    img = cv2.putText(img, string, pt, cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, clr, 1)
    return img

def sortByColumn(arr, column=0, flipped=False):
    if flipped:
        return arr[np.flip(np.argsort(arr[:,0,column]),0)]
    else:
        return arr[np.argsort(arr[:,0,column])]
    
def rotate90(frame):
    frame = cv2.transpose(frame)
    frame = cv2.flip(frame, 1)
    return frame

cap = cv2.VideoCapture(1)
#cap = cv2.VideoCapture("/Users/alexanderwarren/Downloads/cubevid.MOV")
lowerBound = np.array([10, 133, 60])
upperBound = np.array([180, 255, 255])

if (len(sys.argv) > 1):
    if sys.argv[1] == '-m':
        lowerBound = np.array([18, 0, 194])
        upperBound = np.array([42, 255, 255])
#frame_counter = 0
while (True):
    ret, img = cap.read()

    #frame_counter += 1
    #If the last frame is reached, reset the capture and the frame_counter
    #if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT) - 10:
    #    frame_counter = 0 #Or whatever as long as it is the same as next line
    #    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    #img = rotate90(img)
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
        img = cv2.circle(img, tuple(approxPoint[0]), 4, (0, 0, 255), -1)
        text(img, str(i + 1), tuple(approxPoint[0]), (0, 0, 0))
        pass

    topPoint = tuple(contour[contour[:, :, 1].argmin()][0])
    rightPoint = tuple(contour[contour[:, :, 0].argmax()][0])
    leftPoint = tuple(contour[contour[:, :, 0].argmin()][0])
    botPoint = tuple(contour[contour[:, :, 1].argmax()][0])

    midHeight = findHeight(leftPoint, topPoint, rightPoint)
    midHeight = int(midHeight)
    # print(midHeight)
    midPoint = tuple([botPoint[0], topPoint[1] + 2 * midHeight])

    facingHeadon = False
    slightlyHeadon = False
    slightlyHeadonDirection = Direction.LEFT
    if len(approx) <= 4 and len(approx) > 1:
        # for i, pt1 in enumerate(approx):
        #     point1 = pt1.ravel()
        #     for j, pt2 in enumerate(approx):
        #         point2 = pt2.ravel()
        #         if abs(point2[1] - point1[1]) < headOnThresh:
        #             facingHeadon = True
        sortedArr = sortByColumn(approx, column=1, flipped=False)
        if abs(sortedArr[0].ravel()[1] - sortedArr[1].ravel()[1]) < headOnThresh:
            facingHeadon = True
    elif len(approx) == 5:
        sortedArr = sortByColumn(approx, column=1, flipped=False)
        if abs(sortedArr[0].ravel()[1] - sortedArr[1].ravel()[1]) < slightlyHeadonThresh :
            slightlyHeadon = True

    if facingHeadon:
        sortedTop = sortByColumn(approx, column=1, flipped=False)[:2]
        sortedLeftTop = sortByColumn(sortedTop, column=0)

        midPoint = leftPoint
        leftPoint = tuple(sortedLeftTop[0].ravel())
        topPoint = tuple(sortedLeftTop[1].ravel())

    
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


    print(facingHeadon, slightlyHeadon)
    if slightlyHeadon:
        print(slightlyHeadonDirection.name)
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
    print(eulerAngles)
    distance = np.sqrt(tvecs[0].ravel() ** 2 + tvecs[1].ravel() ** 2 + tvecs[2].ravel() ** 2)
    text(img, "distance: %.2f ft." % (distance), (100, 100), (0, 0, 0))

    sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    eulerY = np.arctan2(-rmat[2, 0], sy)
    print((eulerY * (180/np.pi)) % 90)
    res = cv2.bitwise_and(img,img, mask =close2)

    source = drawPnPAxes(source, points, imgpts)
    cv2.imshow("frame2", source)
    cv2.imshow("compound", res)
    k = cv2.waitKey(5)
    if k == 27:
        break
cv2.destroyAllWindows()
