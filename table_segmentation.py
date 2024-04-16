import cv2
import numpy as np

from PIL import Image
from cv2 import Mat

from pic import Pic


def create_pairs(arr):
    return list(zip(arr[:-1], arr[1:]))


def find_morphological_contours(thresh, kernel="horizontal"):
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (40, 1)
    )  # Detect horizontal lines
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, 10)
    )  # Detect vertical lines

    kernel = horizontal_kernel if kernel == "horizontal" else vertical_kernel

    detected = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours = cv2.findContours(detected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours


def thresholding(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh


def crop_image(contour, image):
    x, y, w, h = cv2.boundingRect(contour)
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image[y : y + h, x : x + w]


def find_biggest_contour(binary_image):
    cnts = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
        # cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    contour = max(cnts, key=cv2.contourArea)
    return contour


class TableSegmentation:
    def __init__(self, image: Mat):
        self.img = image
        self.pic = Pic(self.img)
        self.gray = self.pic.gray
        self.binary = self.pic.binary
        self.thresh = thresholding(self.gray)
        self.biggest_contour = find_biggest_contour(self.binary)
        self.detection_color = (136, 201, 3)

    @classmethod
    def make_image_from_image_buffer(self, img_file_buffer):
        image = Image.open(img_file_buffer)
        img_array = np.array(image)
        return img_array

    def crop_image_with_biggest_contour(self, image):
        return crop_image(self.biggest_contour, image)

    def horizontal_lines(self, mask):
        detected_horizontal_contours = find_morphological_contours(
            self.thresh, "horizontal"
        )

        for c in detected_horizontal_contours:
            cv2.drawContours(mask, [c], -1, self.detection_color, 2)
        return mask

    def vertical_lines(self, mask):
        detected_vertical_contours = find_morphological_contours(
            self.thresh, "vertical"
        )

        for c in detected_vertical_contours:
            cv2.drawContours(mask, [c], -1, self.detection_color, 2)
        return mask

    def find_and_crop_biggest_region(self, image=None) -> Mat:
        if image is None:
            image = self.binary

        return crop_image(self.biggest_contour, image)

    def get_horizontal_contours(self):
        detected_horizontal_contours = find_morphological_contours(
            self.thresh, "horizontal"
        )
        return detected_horizontal_contours

    def segment_rows(self):
        sections = []
        for top, bottom in create_pairs(self.get_horizontal_contours()):
            x1, y1, x2, _ = top.flatten()
            _, _, _, y2 = bottom.flatten()

            sections.insert(0, [y2, y1, x1, x2])
        return sections

    def horizontal_segments(self):
        detected_horizontal_contours = self.get_horizontal_contours()
        image = self.img.copy()

        segments = []
        cropped_segments = []

        for index, c in enumerate(detected_horizontal_contours):
            x1, y1, x2, y2 = c.flatten()
            cropped_segment = image[1:y2, x1 : x2 - 2]
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
