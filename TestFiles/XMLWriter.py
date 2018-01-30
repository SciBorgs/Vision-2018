import os
import cv2
from lxml import etree
import xml.etree.cElementTree as ET


def writeXML(folder, img, objects, topLeft, botRight, savedir):
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    image = cv2.imread(img.path)
    height, width, channels = image.shape

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = folder
    ET.SubElement(annotation, "filename").text = img.name
    ET.SubElement(annotation, "segmented").text = "0"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(channels)

    for obj, topL, botR in zip(objects, topLeft, botRight):
        ob = ET.SubElement(annotation, "object")
        ET.SubElement(ob, "name").text = obj
        ET.SubElement(ob, "pose").text = "Unspecified"
        ET.SubElement(ob, "truncated").text = "0"
        ET.SubElement(ob, "difficult").text = "0"

        bndBox = ET.SubElement(ob, "bndbox")
        ET.SubElement(bndBox, "xmin").text = str(topL[0])
        ET.SubElement(bndBox, "ymin").text = str(topL[1])
        ET.SubElement(bndBox, "xmax").text = str(botR[0])
        ET.SubElement(bndBox, "ymax").text = str(botR[1])

    xmlStr = ET.tostring(annotation)
    root = etree.fromstring(xmlStr)
    xmlStr = etree.tostring(root, pretty_print=True)
    savePath = os.path.join(savedir, img.name.replace("png", "xml"))

    with open(savePath, "wb") as tempXml:
        tempXml.write(xmlStr)

#
# if __name__ == '__main__':
#     folder = "Both"
#     imgs = [im for im in os.scandir("Both") if ""]
