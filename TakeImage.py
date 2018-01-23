import cv2
import sys

for x in range(1, 5):
    stream = cv2.VideoCapture(x)

    if (stream.isOpened()):
        print("Camera found on port: %d" % (x))
        break;

if (not stream.isOpened()):
    print("Camera not found")
    sys.exit()

n = 0

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
