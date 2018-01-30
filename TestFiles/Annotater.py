import cv2
import os
import matplotlib.pylab as plt
from matplotlib.widgets import RectangleSelector
from XMLWriter import writeXML

img = None
topLeftList = []
topLeft = None
botRightList = []
botRight = None
objects = []

folder = "Both"
saveDir = "Annotations"
label = "Power_Cubes"


def lineSelectCallback(clk, rls):
    global topLeft
    global botRight

    topLeft = (int(clk.xdata), int(clk.ydata))
    botRight = (int(rls.xdata), int(rls.ydata))


def onKeyPress(event):
    global objects
    global topLeftList
    global botRightList
    global topLeft
    global botRight
    global img

    if event.key == "q":
        writeXML(folder, img, objects, topLeftList, botRightList, saveDir)

        topLeftList = []
        botRightList = []
        objects = []
        img = None
        plt.close()

    if event.key == "c":
        print("Added coords: {} | {}".format(topLeft, botRight))
        topLeftList.append(topLeft)
        botRightList.append(botRight)
        objects.append(label)


def toggleSelector(event):
    toggleSelector.RS.set_active(True)


if __name__ == '__main__':
    for n, file in enumerate(os.scandir(folder)):
        img = file

        fig, ax = plt.subplots(1)

        image = cv2.imread(file.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)

        toggleSelector.RS = RectangleSelector(ax,
                                              lineSelectCallback,
                                              drawtype="box",
                                              useblit=True,
                                              button=[1],
                                              minspanx=5,
                                              minspany=5,
                                              spancoords="pixels",
                                              interactive=True)

        bndBox = plt.connect("key_press_event", toggleSelector)
        key = plt.connect("key_press_event", onKeyPress)
        plt.show()
