import cv2
from Vision.DistAngleVision import DistAngleVision
import numpy as np
from Vision.NetworkTableHandler import NetworkTableHandler
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':

    for x in range(1, 5):
        stream = cv2.VideoCapture(x)

        if (stream.isOpened()):
            print("Camera found on port: %d" % (x))
            break

    if (not stream.isOpened()):
        print("Camera not found")
        sys.exit()

    lowerBound = np.array([17, 163, 70])
    upperBound = np.array([30, 219, 227])

    ntHandler = NetworkTableHandler()

    # Lifecam is 60 degrees from left to right. Pass it only half of fov
    vision = DistAngleVision(stream.get(cv2.CAP_PROP_FRAME_WIDTH), stream.get(cv2.CAP_PROP_FRAME_HEIGHT), 30, ntHandler)

    if (len(sys.argv) > 1):
        if sys.argv[1].find('n') != -1:
            vision.lowerBound = ntHandler.getHSVValues("lower")
            vision.upperBound = ntHandler.getHSVValues("upper")

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
