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

        # Detect horizontal lines
        self.horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))

        # Detect vertical lines
        self.vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))

    def morphological_contours(self, kernel):
        detected = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        return cv2.findContours(detected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def thresholding(self):
        img = self.gray.copy()
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    def empty_mask(self):
        return np.zeros(self.img.shape, dtype=np.uint8)
        # return cv2.cvtColor(np.zeros(self.gray.shape), cv2.COLOR_GRAY2BGR)

    def horizontal_lines(self, mask):
        detect_horizontal_contours = self.morphological_contours(self.horizontal_kernel)
        cnts = detect_horizontal_contours
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        color = (36, 255, 12)  # green
        for c in cnts:
            cv2.drawContours(mask, [c.astype(int)], -1, color, 2)
        return mask

    def vertical_lines(self, mask):
        detect_vertical_contours = self.morphological_contours(self.vertical_kernel)
        cnts = detect_vertical_contours[0] if len(detect_vertical_contours) == 2 else detect_vertical_contours[1]
        color = (0, 0, 255)  # blue
        for c in cnts:
            cv2.drawContours(mask, [c], -1, color, 2)
        return mask

    def horizontal_segments(self):
        detect_horizontal_contours = self.morphological_contours(self.horizontal_kernel)
        image = self.img.copy()
        cnts = detect_horizontal_contours[0] if len(detect_horizontal_contours) == 2 else detect_horizontal_contours[1]

        color = (36, 255, 12)  # green

        segments = []
        cropped_segments = []

        for index, c in enumerate(cnts):
            x1, y1, x2, y2 = c.flatten()
            cropped_segment = image[1:y2, x1:x2 - 2]
            if index % 2 == 1:
                continue
            cropped_segments.append(cropped_segment)
            #
            # segments.append((x1, y1, x2, y2))
            # se

        #
        # cropped_segments = [segments[i:i + 2] for i in range(0, len(segments), 2)]
        # iii = []
        #
        # for i in cropped_segments:
        #     top, bottom = i
        #     cropped_segment = image[top:bottom, 0:500]
        #     iii.append(cropped_segment)
        #
        # #     cv2.drawContours(image, [c.astype(int)], -1, color, 2)
        # #     x, y, w, h = cv2.boundingRect(c)
        # #     cropped_segment = image[y:y+h, x:x+w]
        # #     cropped_segments.append(cropped_segment)
        #
        return cropped_segments
