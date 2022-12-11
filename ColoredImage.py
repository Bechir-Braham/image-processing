import numpy as np

from GrayscaleImage import GrayscaleImage


class ColoredImage:
    def __init__(self):
        self.lines = 0
        self.cols = 0
        self.data = []
        self.r = None
        self.g = None
        self.b = None

    def read_from_array(self, data):
        self.r = data[:, :, 0]
        self.g = data[:, :, 1]
        self.b = data[:, :, 2]
        self.data = data
        self.lines = data.shape[0]
        self.cols = data.shape[1]
        return self

    def apply_equalization(self):
        r, g, b = GrayscaleImage(), GrayscaleImage(), GrayscaleImage()
        r.read_from_array(self.r)
        g.read_from_array(self.g)
        b.read_from_array(self.b)
        r.apply_map(r.equalization_array())
        g.apply_map(g.equalization_array())
        b.apply_map(b.equalization_array())
        r = np.array(r.data)
        g = np.array(g.data)
        b = np.array(b.data)

        tmp = np.array([r, g, b]).reshape((3, r.shape[0], r.shape[1]))
        self.data = np.moveaxis(tmp, 0, -1)
        self.r = self.data[:, :, 0]
        self.g = self.data[:, :, 1]
        self.b = self.data[:, :, 2]
        return self

    def apply_linear_transformation(self, p1x, p1y, p2x, p2y):
        r, g, b = GrayscaleImage(), GrayscaleImage(), GrayscaleImage()
        r.read_from_array(self.r)
        g.read_from_array(self.g)
        b.read_from_array(self.b)
        map = r.piecewise_linear((p1x, p1y), (p2x, p2y))
        r.apply_map(map)
        g.apply_map(map)
        b.apply_map(map)
        r = np.array(r.data)
        g = np.array(g.data)
        b = np.array(b.data)
        tmp = np.array([r, g, b]).reshape((3, r.shape[0], r.shape[1]))
        self.data = np.moveaxis(tmp, 0, -1)
        self.r = self.data[:, :, 0]
        self.g = self.data[:, :, 1]
        self.b = self.data[:, :, 2]
        return map

    def apply_threshold(self, thresh_r, thresh_g, thresh_b):
        self.r[self.r > thresh_r] = 255
        self.r[self.r <= thresh_r] = 0
        self.g[self.g > thresh_g] = 255
        self.g[self.g <= thresh_g] = 0
        self.b[self.b > thresh_b] = 255
        self.b[self.b <= thresh_b] = 0
        return self
