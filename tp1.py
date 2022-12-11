class Image:
    def __init__(self):
        self.format = ''
        self.lines = 0
        self.cols = 0
        self.pixels = 0
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
        ret = [tmp]
        for i in self.hist:
            tmp += i
            ret.append(tmp)
        return ret


def main():
    img = Image()
    img.read('./chat.pgm')
    img.write('saroura.pgm')
    print('mean', img.moment(1))
    print('std', img.std())
    print('histogram', img.histogram())
    print('cumulative histogram', img.cumul_histogram())

    # d=np.array(img.data)  # print(d.mean())  # print(d.std())


if __name__ == '__main__':
    main()
