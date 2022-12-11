class GrayscaleImage:
    def __init__(self):
        self.format = ''
        self.lines = 0
        self.cols = 0
        self.pixels = 255
        self.data = []
        self.comment = ''
        self.hist = []

    def read(self, path):
        with open(path) as f:
            self.format = f.readline()[:-1]
            if self.format == 'P2':
                self.read_pgm(f)
            elif self.format == 'P3':
                self.read_ppm(f)

    def read_ppm(self, f):
        pass

    def read_from_array(self, arr):
        self.lines = arr.shape[0]
        self.cols = arr.shape[1]
        self.pixels = 255
        self.data = arr

    def read_pgm(self, f):
        self.comment = f.readline()[:-1]
        tmp = f.readline()
        self.cols, self.lines = tmp.split()
        self.cols, self.lines = int(self.cols), int(self.lines)
        self.pixels = int(f.readline())
        self.data = []
        for line in range(self.lines):
            tmp = f.readline()
            tmp = tmp.split()
            tmp = [int(x) for x in tmp]
            self.data.append(tmp)

        return self.data

    def write(self, path):
        with open(path, 'w') as f:
            f.write(self.format)
            f.write('\n')
            f.write(self.comment)
            f.write('\n')
            f.write(str(self.cols) + " " + str(self.lines))
            f.write('\n')
            f.write(str(self.pixels))
            f.write('\n')
            for line in self.data:
                tmp = [str(x) for x in line]
                f.write(" ".join(tmp))
                f.write('\n')

    def moment(self, power):
        tmp = 0
        for line in self.data:
            for i in line:
                tmp += i ** power
        return tmp / (self.lines * self.cols)

    def std(self):
        return (self.moment(2) - self.moment(1) ** 2) ** 0.5

    def histogram(self):
        freq_arr = [0] * 256
        for line in self.data:
            for i in line:
                freq_arr[i] += 1
        self.hist = freq_arr
        return freq_arr

    def cumul_histogram(self):
        self.histogram()
        tmp = 0
        ret = []
        for i in self.hist:
            tmp += i
            ret.append(tmp)
        return ret

    def equalization_array(self):
        cumul = self.cumul_histogram()
        cumul_std = (i / (self.cols * self.lines) for i in cumul)
        cumul_eg = (i * self.pixels for i in cumul_std)
        ret = [int(i) for i in cumul_eg]
        return ret

    def apply_map(self, map):
        tmp = []
        for line in self.data:
            tmp.append([map[i] for i in line])
        self.data = tmp

    def piecewise_linear(self, point1, point2):
        map = []
        for i in range(point1[0] ):
            rate = point1[1] / point1[0]
            map.append(int(rate * i))

        for i in range(point1[0] , point2[0] + 1):
            rate = (point2[1] - point1[1]) / (point2[0] - point1[0])
            offset = point2[1] - rate * point2[0]
            map.append(int(rate * i + offset))
        point3 = (255, 255)

        for i in range(point2[0] , point3[0] + 1):
            rate = (point3[1] - point2[1]) / (point3[0] - point2[0])
            offset = point3[1] - rate * point3[0]
            map.append(int(rate * i + offset))
        return map

    def equalized_cumulative(self):
        eq = self.equalization_array()
        cumul = self.cumul_histogram()
        map = list(range(256))
        for i in eq:
            map[i] += cumul[i]
        return map
