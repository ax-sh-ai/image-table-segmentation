import cv2
import imutils
import base64
from cv2 import Mat
from IPython.display import display
from PIL import Image
from collections import namedtuple
import numpy as np

Thumb = namedtuple('Thumb', ['label', 'mat'])


# Define a subclass of ndarray
class JupyterMat(np.ndarray):
    @staticmethod
    def data_url(mat: np.ndarray):
        retval, buffer = cv2.imencode('.jpg', mat)
        return f'data:image/png;base64,{base64.b64encode(buffer).decode()}'

    # uses nparray.view(JupyterMat)
    # uses mat.view(JupyterMat)
    def _repr_html_(self):
        resized = imutils.resize(self, width=300)
        url = self.data_url(resized)
        return f'<img src="{url}"/>'


class MatArray(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    @staticmethod
    def data_url(mat: np.ndarray):
        retval, buffer = cv2.imencode('.jpg', mat)
        return f'data:image/png;base64,{base64.b64encode(buffer).decode()}'

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def _repr_html_(self):
        """
        Return an HTML representation of the array.
        """
        resized = imutils.resize(self, width=300)
        url = self.data_url(resized)
        return f'<img src="{url}"/>'


class Pic:
    def __init__(self, image: np.ndarray):
        # if isinstance(image, None):
        #     raise TypeError('image must be an instance of Mat')
        self.image = image
        # self.shape = self.image.shape

        self.mats: list[Thumb] = []

        self._add_to_mat("BGR", self.image)

        # self.add_to_mat(self.gray)
        self.thumb_size = 300
        self.rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def mask(self):
        return MatArray(np.zeros(self.gray, np.uint8))

    @property
    def gray(self):
        return MatArray(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY))

    def invert(self):
        return np.invert(self.gray)

    def resize(self, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim: tuple = tuple()
        (h, w) = self.image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return self.image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(self.image, dim, interpolation=inter)

        # return the resized image
        return resized

    def _add_to_mat(self, label: str, mat: np.ndarray):
        im = imutils.resize(mat, width=500)
        self.mats.append(Thumb(label, im))

    def __repr__(self):
        height, width = self.image.shape[:2]
        # , channels)
        return f'<Pic(height={height}, width={width})>'

    def _repr_html_(self):

        urls = [(thumb.label, self.data_url(thumb.mat)) for thumb in self.mats]
        images = [f'<div><h6>{label}</h6><img src="{url}"/></div>' for label, url in urls]
        images = "\n".join(images)
        return (f'<section>'
                '<h6>Pic Images</h6>'
                f'<div style="display:flex;gap:2rem;flex-wrap:wrap">'
                f"{images}"
                f'</div>'
                f'</section>')

    @staticmethod
    def data_url(mat: Mat):
        retval, buffer = cv2.imencode('.jpg', mat)
        return f'data:image/png;base64,{base64.b64encode(buffer).decode()}'

    @staticmethod
    def show(image: Mat):
        display(Image.fromarray(image))
