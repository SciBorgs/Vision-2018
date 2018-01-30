import cv2
import sys
import os
from random import randint

os.system("v4l2-ctl -d /dev/video1"
          " -c brightness={}".format(10 + randint(-5, 5)) +
          " -c white_balance_temperature_auto=false"
          " -c exposure_auto=1"
          )

for x in range(1, 5):
    stream = cv2.VideoCapture(x)

    if (stream.isOpened()):
        print("Camera found on port: %d" % (x))
        break

if (not stream.isOpened()):
    print("Camera not found")
    sys.exit()

n = 61

while True:

    ret, source = stream.read()

    cv2.imshow("source", source)
    keyPressed = cv2.waitKey(33)

    if (keyPressed == ord("s")):
        cv2.imwrite("{:6}.png".format(n), source)
        print(n)
        n += 1
    elif (keyPressed == ord("q")):
        cv2.destroyAllWindows()
        sys.exit()
