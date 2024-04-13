import cv2
import numpy as np

from PIL import Image
from pic import Pic


class TableSegmentation:
    def __init__(self):
        self.img = cv2.imread('table.png')
        self.pic = Pic(self.img)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def thresholding(self):
        img = self.gray.copy()
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    def horizontal_lines(self):
        result = self.img.copy()
        thresh = self.thresholding()
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        color = (36, 255, 12)
        for c in cnts:
            cv2.drawContours(result, [c], -1, color, 2)
        return result

    def vertical_lines(self):
        result = self.img.copy()
        thresh = self.thresholding()
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
        detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(result, [c], -1, (36,255,12), 2)
        return result