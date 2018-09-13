from networktables import NetworkTables as nt
import numpy as np


class NetworkTableHandler:
    # Class room default
    # lowerBound = np.array([17, 163, 70])
    # upperBound = np.array([30, 219, 227])

    # Staircase default
    # lowerBound = np.array([0, 186, 64])
    # upperBound = np.array([180, 255, 255])

    # Yoga room default
    lowerBound = np.array([18, 148, 107])
    upperBound = np.array([31, 255, 219])

    def __init__(self):
        nt.initialize(server="roboRIO-1155-frc.local")
        self.sd = nt.getTable("SmartDashboard")

    def setValue(self, key, value):
        self.sd.putNumber(key, value)

    def getValue(self, key):
        return self.sd.getNumber(key, 'N/A')

    def getHSVValues(self, bound):
        if bound == "lower":
            if (self.getValue("Lower Hue") is not None) and (self.getValue("Lower Saturation") is not None) and (self.getValue("Lower Value") is not None):
                return np.array([int(self.getValue("Lower Hue")), int(self.getValue("Lower Saturation")), int(self.getValue("Lower Value"))])
            else:
                return NetworkTableHandler.lowerBound
        elif bound == "upper":
            if (self.getValue("Upper Hue") is not None) and (self.getValue("Upper Saturation") is not None) and (self.getValue("Upper Value") is not None):
                return np.array([int(self.getValue("Upper Hue")), int(self.getValue("Upper Saturation")), int(self.getValue("Upper Value"))])
            else:
                return NetworkTableHandler.upperBound
