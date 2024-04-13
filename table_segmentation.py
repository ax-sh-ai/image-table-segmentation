import cv2
import numpy as np

from PIL import Image


class TableSegmentation:
    def __init__(self):
        self.img = cv2.imread('table.png')
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
