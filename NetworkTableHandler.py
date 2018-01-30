from networktables import NetworkTables as nt

class NetworkTableHandler:

	def __init__(self):
		nt.initialize(server="roboRIO-1155-frc.local")
		self.sd = nt.getTable("SmartDashboard")

	def setValue(self, key, value):
		self.sd.putNumber(key, value)

	def getValue(self, key):
		return self.sd.getNumber(key, 'N/A')

