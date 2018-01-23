class CScore:

    def __init__(self, contour, size, extent, solidity):
        self.contour = contour
        self.size = size
        self.extent = extent
        self.solidity = solidity

    def getSize(self):
        return self.size

    def getExtent(self):
        return self.extent

    def getSolidity(self):
        return self.solidity

    def getScore(self):

