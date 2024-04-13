import cv2
import numpy as np

from PIL import Image
from pic import Pic


class TableSegmentation:
    def __init__(self):
        self.img = cv2.imread('table.png')
        self.pic = Pic(self.img)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.thresh = self.thresholding()

    def thresholding(self):
        img = self.gray.copy()
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    def empty_mask(self):
        return np.zeros(self.img.shape, dtype=np.uint8)
        # return cv2.cvtColor(np.zeros(self.gray.shape), cv2.COLOR_GRAY2BGR)

    def horizontal_lines(self, mask):

        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detect_horizontal = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        color = (36, 255, 12)
        print(cnts)
        for c in cnts:
            cv2.drawContours(mask, [c.astype(int)], -1, color, 2)
        return mask

    def vertical_lines(self, mask):
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
        detect_vertical = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        color = (0, 0, 255)
        for c in cnts:
            cv2.drawContours(mask, [c], -1, color, 2)
        return mask