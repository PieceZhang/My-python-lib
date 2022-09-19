import numpy as np


class Curve(object):
    """
    保存一段曲线，并在buf溢出后从第一位从头存储
    """
    def __init__(self, maxl=100):
        self.maxl = maxl
        self.data = np.ndarray([maxl])
        self.count = 0
        self.ifover = False

    def append(self, newdata):
        self.data[self.count] = newdata
        self.count += 1
        if self.count >= self.maxl:
            self.count = 0
            self.ifover = True

    def readline(self):
        if self.ifover:
            return np.concatenate([self.data[self.count:], self.data[:self.count]])
        else:
            return self.data[:self.count]

    def readpoint(self):
        return self.data[self.count - 1]
