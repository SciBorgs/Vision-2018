from networktables import NetworkTables as nt

class NetworkTableHandler:

	def __init__(self):
		nt.initialize(server="roboRIO-1155-frc.local")
		self.sd = nt.getTable("SmartDashboard")

	def setValue(self, key, value):
		self.sd.putNumber(key, value)

	def getValue(self, key):
		return self.sd.getNumber(key, 'N/A')

	def getHSVValues(self, bound):
		if bound == "lower":
			if (self.getValue("Lower Hue") != None) && (self.getValue("Lower Saturation") != None) && (self.getValue("Lower Value") != None):
				return np.array([int(self.getValue("Lower Hue")), int(self.getValue("Lower Saturation")), int(self.getValue("Lower Value"))])
		elif bound == "upper":
			if (self.getValue("Upper Hue") != None) && (self.getValue("Upper Saturation") != None) && (self.getValue("Upper Value") != None):
				return np.array([int(self.getValue("Upper Hue")), int(self.getValue("Upper Saturation")), int(self.getValue("Upper Value"))])