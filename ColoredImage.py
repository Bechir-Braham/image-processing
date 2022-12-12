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

        self.data = data.copy()
        self.r = self.data[:, :, 0]
        self.g = self.data[:, :, 1]
        self.b = self.data[:, :, 2]
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

    def get_histogram(self, arr):
        freq_arr = [0] * 256
        for line in arr:
            for i in line:
                freq_arr[i] += 1
        return freq_arr

    def get_cumul_histogram(self, arr):
        hist = self.get_histogram(arr)
        tmp = 0
        ret = []
        for i in hist:
            tmp += i
            ret.append(tmp)
        return ret

    def get_Otsu_segmentation_value(self, hist):
        """
        :param hist: takes in a histogram as a parameter
        :return: returns an integer value between 0-255
        """
        values = []
        for value in range(256):
            count_less = 0
            sum_less = 0
            count_more = 0
            sum_more = 0
            for pixel_value in range(len(hist)):
                if pixel_value <= value:
                    count_less += hist[pixel_value]
                    sum_less += hist[pixel_value] * pixel_value
                else:
                    count_more += hist[pixel_value]
                    sum_more += hist[pixel_value] * pixel_value
            pixels = sum(hist)
            weight_less = count_less / pixels
            weight_more = count_more / pixels

            mean_less = sum_less / count_less if count_less != 0 else 0
            mean_more = sum_more / count_more if count_more != 0 else 0
            metric = weight_less * weight_more * (mean_less - mean_more) ** 2
            values.append((metric, value))
        return max(values)[1]

    def apply_Otsu_segmentation(self):
        val_r = self.get_Otsu_segmentation_value(self.get_histogram(self.r))
        val_g = self.get_Otsu_segmentation_value(self.get_histogram(self.g))
        val_b = self.get_Otsu_segmentation_value(self.get_histogram(self.b))
        self.apply_threshold(val_r, val_g, val_b)
        return val_r, val_g, val_b

    def add_noise(self):
        noise = np.random.randint(21, size=(self.lines, self.cols))
        for i in range(self.lines):
            for j in range(self.cols):
                tmp = noise[i, j]
                if tmp == 0:
                    self.data[i, j] = [0, 0, 0]
                elif tmp == 20:
                    self.data[i, j] = [255, 255, 255]

        return self

    def apply_filter_grayscale(self, matrix, data):
        n = 3  # given that the matrix is nxn
        new_data = np.zeros_like(data)
        for i in range(self.lines):
            for j in range(self.cols):
                rstart = i - n // 2
                rend = i + n // 2
                cstart = j - n // 2
                cend = j + n // 2
                if rstart < 0 or rend >= self.lines or cstart < 0 or cend >= self.cols:
                    continue
                block = data[rstart:rend + 1, cstart:cend + 1]
                pixel = np.sum(block * matrix)
                new_data[i, j] = int(np.clip(pixel, 0, 255))
        np.copyto(data, new_data)
        return data

    def apply_median_filter_grayscale(self, data):
        n = 3  # given that the matrix is nxn
        new_data = np.zeros_like(data)
        for i in range(self.lines):
            for j in range(self.cols):
                rstart = i - n // 2
                rend = i + n // 2
                cstart = j - n // 2
                cend = j + n // 2
                if rstart < 0 or rend >= self.lines or cstart < 0 or cend >= self.cols:
                    continue
                block = data[rstart:rend + 1, cstart:cend + 1]
                new_data[i, j] = np.median(block)
        np.copyto(data, new_data)
        return data

    def apply_median(self):
        self.apply_median_filter_grayscale(self.r)
        self.apply_median_filter_grayscale(self.g)
        self.apply_median_filter_grayscale(self.b)
        return self

    def apply_filter(self, matrix):
        self.apply_filter_grayscale(matrix, self.r)
        self.apply_filter_grayscale(matrix, self.g)
        self.apply_filter_grayscale(matrix, self.b)
        return self

    def get_three_histograms(self):
        r = self.get_histogram(self.r)
        g = self.get_histogram(self.g)
        b = self.get_histogram(self.b)
        return r, g, b
