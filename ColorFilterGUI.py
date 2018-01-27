import cv2
import wx
import numpy as np
import threading
import sys
import os
from random import randint


class ColorFilterClass(wx.Frame):

    def __init__(self, *args, **kwargs):
        super(ColorFilterClass, self).__init__(*args, **kwargs)

        os.system("v4l2-ctl -d /dev/video1"
                  " -c brightness={}".format(80 + randint(-5, 5)) +
                  " -c white_balance_temperature_auto=false"
                  " -c exposure_auto=1")

        self.initUI()

    def startVision(self):

        for x in range(1, 5):
            stream = cv2.VideoCapture(x)

            if (stream.isOpened()):
                print("Camera found on port: %d" % (x))
                break;

        if (not stream.isOpened()):
            print("Camera not found")
            sys.exit()

        while True:

            LOWER_LIMIT = np.array([self.lSlider1.GetValue(), self.lSlider2.GetValue(), self.lSlider3.GetValue()])
            UPPER_LIMIT = np.array([self.uSlider1.GetValue(), self.uSlider2.GetValue(), self.uSlider3.GetValue()])

            rc, source = stream.read()
            original = source

            if (self.sliderOption == "HSV"):
                source = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(source, LOWER_LIMIT, UPPER_LIMIT)

            erode = cv2.erode(mask, kernel=self.createKernel(3))
            dilate = cv2.dilate(erode, kernel=self.createKernel(3))
            close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, self.createKernel(15))
            erode2 = cv2.dilate(close, kernel=self.createKernel(7))

            cv2.imshow("Chain Morphs", erode2)

            fStream, validContours, hierarchy = cv2.findContours(erode2.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

            for c in validContours:

                try:
                    cMoments = cv2.moments(c)
                    centerPoint = (int((cMoments["m10"] / cMoments['m00'])),
                                   int((cMoments["m01"] / cMoments["m00"])))
                    EPSILON = (self.epsSlider.GetValue() / 100) * cv2.arcLength(c, True)

                    c = c.astype('int')
                    cArea = cv2.contourArea(c)
                    cPerimeter = cv2.arcLength(c, True)
                    cBoundary = cv2.approxPolyDP(c, EPSILON, True)
                    x, y, w, h = cv2.boundingRect(cBoundary)

                    cv2.drawContours(original, [cBoundary], -1, [0, 255, 0], 3)

                except ZeroDivisionError:
                    pass

            cv2.imshow("Stream", original)

            keyPressed = cv2.waitKey(33)

            if keyPressed == ord("q"):
                cv2.destroyAllWindows()
                sys.exit()
            elif keyPressed == ord("s"):
                cv2.imwrite("Test.jpg", source)

    def initUI(self):
        panel = wx.Panel(self)
        # On start up selection
        self.sliderOption = "HSV"

        mainSizer = wx.BoxSizer(wx.VERTICAL)

        # Creating title to be shown on GUI
        guiTitle = wx.StaticText(panel, label="HSV and BGR Color Filtering GUI")
        font = guiTitle.GetFont()
        font.SetWeight(wx.BOLD)
        font.SetUnderlined(True)
        font.SetPointSize(18)
        guiTitle.SetFont(font)

        # Centering title with slight buffer zone on top
        mainSizer.AddStretchSpacer(prop=1)
        mainSizer.Add(guiTitle, 0, wx.CENTER)

        # Setting up a Horizontal Sizer for the radio buttons
        mainSizer.AddStretchSpacer(prop=1)
        rbSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.hsvRB = wx.RadioButton(panel, label="HSV", style=wx.RB_GROUP)
        self.bgrRB = wx.RadioButton(panel, label="BGR")
        self.Bind(wx.EVT_RADIOBUTTON, self.swapSliderRange)

        rbSizer.AddStretchSpacer(prop=1)
        rbSizer.Add(self.hsvRB, 0, wx.CENTER)
        rbSizer.AddStretchSpacer(prop=1)
        rbSizer.Add(self.bgrRB, 0, wx.CENTER)
        rbSizer.AddStretchSpacer(prop=1)
        mainSizer.Add(rbSizer, 0, wx.EXPAND)

        # Setting up Lower Slider rows
        mainSizer.AddStretchSpacer(prop=1)
        l_SSizer1 = wx.BoxSizer(wx.HORIZONTAL)
        self.lSlider1Text = wx.StaticText(panel, label="Lower Hue")
        self.lSlider1 = wx.Slider(panel, value=0, minValue=0, maxValue=180, size=(600, -1), style=wx.SL_AUTOTICKS | wx.SL_LABELS)

        l_SSizer1.AddStretchSpacer(prop=1)
        l_SSizer1.Add(self.lSlider1Text, 0, wx.CENTER)
        l_SSizer1.AddStretchSpacer(prop=1)
        l_SSizer1.Add(self.lSlider1, 0, wx.CENTER)
        l_SSizer1.AddStretchSpacer(prop=1)
        mainSizer.Add(l_SSizer1, 0, wx.EXPAND)

        mainSizer.AddStretchSpacer(prop=1)
        l_SSizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.lSlider2Text = wx.StaticText(panel, label="Lower Sat")
        self.lSlider2 = wx.Slider(panel, value=0, minValue=0, maxValue=255, size=(600, -1), style=wx.SL_AUTOTICKS | wx.SL_LABELS)

        l_SSizer2.AddStretchSpacer(prop=1)
        l_SSizer2.Add(self.lSlider2Text, 0, wx.CENTER)
        l_SSizer2.AddStretchSpacer(prop=1)
        l_SSizer2.Add(self.lSlider2, 0, wx.CENTER)
        l_SSizer2.AddStretchSpacer(prop=1)
        mainSizer.Add(l_SSizer2, 0, wx.EXPAND)

        mainSizer.AddStretchSpacer(prop=1)
        l_SSizer3 = wx.BoxSizer(wx.HORIZONTAL)
        self.lSlider3Text = wx.StaticText(panel, label="Lower Val")
        self.lSlider3 = wx.Slider(panel, value=0, minValue=0, maxValue=255, size=(600, -1), style=wx.SL_AUTOTICKS | wx.SL_LABELS)

        l_SSizer3.AddStretchSpacer(prop=1)
        l_SSizer3.Add(self.lSlider3Text, 0, wx.CENTER)
        l_SSizer3.AddStretchSpacer(prop=1)
        l_SSizer3.Add(self.lSlider3, 0, wx.CENTER)
        l_SSizer3.AddStretchSpacer(prop=1)
        mainSizer.Add(l_SSizer3, 0, wx.EXPAND)

        # Setting up Upper Slider rows
        mainSizer.AddStretchSpacer(prop=5)
        u_SSizer1 = wx.BoxSizer(wx.HORIZONTAL)
        self.uSlider1Text = wx.StaticText(panel, label="Upper Hue")
        self.uSlider1 = wx.Slider(panel, value=180, minValue=0, maxValue=180, size=(600, -1), style=wx.SL_AUTOTICKS | wx.SL_LABELS)

        u_SSizer1.AddStretchSpacer(prop=1)
        u_SSizer1.Add(self.uSlider1Text, 0, wx.CENTER)
        u_SSizer1.AddStretchSpacer(prop=1)
        u_SSizer1.Add(self.uSlider1, 0, wx.CENTER)
        u_SSizer1.AddStretchSpacer(prop=1)
        mainSizer.Add(u_SSizer1, 0, wx.EXPAND)

        mainSizer.AddStretchSpacer(prop=1)
        u_SSizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.uSlider2Text = wx.StaticText(panel, label="Upper Sat")
        self.uSlider2 = wx.Slider(panel, value=255, minValue=0, maxValue=255, size=(600, -1), style=wx.SL_AUTOTICKS | wx.SL_LABELS)

        u_SSizer2.AddStretchSpacer(prop=1)
        u_SSizer2.Add(self.uSlider2Text, 0, wx.CENTER)
        u_SSizer2.AddStretchSpacer(prop=1)
        u_SSizer2.Add(self.uSlider2, 0, wx.CENTER)
        u_SSizer2.AddStretchSpacer(prop=1)
        mainSizer.Add(u_SSizer2, 0, wx.EXPAND)

        mainSizer.AddStretchSpacer(prop=1)
        u_SSizer3 = wx.BoxSizer(wx.HORIZONTAL)
        self.uSlider3Text = wx.StaticText(panel, label="Upper Val")
        self.uSlider3 = wx.Slider(panel, value=255, minValue=0, maxValue=255, size=(600, -1), style=wx.SL_AUTOTICKS | wx.SL_LABELS)

        u_SSizer3.AddStretchSpacer(prop=1)
        u_SSizer3.Add(self.uSlider3Text, 0, wx.CENTER)
        u_SSizer3.AddStretchSpacer(prop=1)
        u_SSizer3.Add(self.uSlider3, 0, wx.CENTER)
        u_SSizer3.AddStretchSpacer(prop=1)
        mainSizer.Add(u_SSizer3, 0, wx.EXPAND)

        # Setting up the slider for the contour approximation constant
        mainSizer.AddStretchSpacer(prop=5)
        epsSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.epsText = wx.StaticText(panel, label="Epsilon")
        self.epsSlider = wx.Slider(panel, value=0, minValue=0, maxValue=100, size=(600, -1), style=wx.SL_AUTOTICKS | wx.SL_LABELS)

        epsSizer.AddStretchSpacer(prop=1)
        epsSizer.Add(self.epsText, 0, wx.CENTER)
        epsSizer.AddStretchSpacer(prop=1)
        epsSizer.Add(self.epsSlider, 0, wx.CENTER)
        epsSizer.AddStretchSpacer(prop=1)
        mainSizer.Add(epsSizer, 0, wx.EXPAND)

        mainSizer.AddStretchSpacer(prop=10)

        panel.SetSizer(mainSizer)
        self.Show()

        self.visionThread = threading.Thread(name="Vision Thread", target=self.startVision())
        self.visionThread.start()

    def swapSliderRange(self, event):
        radioButton = event.GetEventObject()

        if (radioButton.GetLabel() == "HSV"):
            self.lSlider1.SetMax(180)
            self.lSlider2.SetMax(255)
            self.lSlider3.SetMax(255)

            self.uSlider1.SetMax(180)
            self.uSlider2.SetMax(255)
            self.uSlider3.SetMax(255)

            self.lSlider1Text.SetLabel("Lower Hue")
            self.lSlider2Text.SetLabel("Lower Sat")
            self.lSlider3Text.SetLabel("Lower Val")

            self.uSlider1Text.SetLabel("Upper Hue")
            self.uSlider2Text.SetLabel("Upper Sat")
            self.uSlider3Text.SetLabel("Upper Val")

            self.sliderOption = "HSV"

        elif (radioButton.GetLabel() == "BGR"):
            self.lSlider1.SetMax(255)
            self.lSlider2.SetMax(255)
            self.lSlider3.SetMax(255)

            self.uSlider1.SetMax(255)
            self.uSlider2.SetMax(255)
            self.uSlider3.SetMax(255)

            self.lSlider1Text.SetLabel("Lower Blue")
            self.lSlider2Text.SetLabel("Lower Green")
            self.lSlider3Text.SetLabel("Lower Red")

            self.uSlider1Text.SetLabel("Upper Blue")
            self.uSlider2Text.SetLabel("Upper Green")
            self.uSlider3Text.SetLabel("Upper Red")

            self.sliderOption = "BGR"

    def createKernel(self, size):
        return np.ones((size, size), np.uint8)


if __name__ == "__main__":
    app = wx.App()
    ColorFilterClass(None, title="Color Filter GUI", size=(800, 800), style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)
    app.MainLoop()
