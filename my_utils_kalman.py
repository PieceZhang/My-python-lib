import numpy as np


class KalmanFilter:
    def __init__(self, Q=1, R=5):
        self.Q = Q
        self.R = R
        self.p_last = 10
        self.x_last = 0

    def predict(self, Z):
        a = 1
        b = 0
        c = 1

        # q = 0
        # r = 0
        # beta = 0.001
        # alpha = 50

        # 预测步
        x_ = a * self.x_last  # 预测当前状态
        p_ = a * self.p_last * a + self.Q
        e = Z - x_  # 计算残差

        # 更新步
        k = p_ * c / (c * p_ * c + self.R)
        x = x_ + k * e
        p = (1 - k * c) * p_

        self.p_last = p
        self.x_last = x

        return x


class SageHusaAdaptiveKalmanFilter:
    def __init__(self, Q=1, R=5, softstartskip=0):
        self.Q = Q
        self.R = R
        self.q = 0
        self.r = 0
        self.p_last = 10
        self.x_last = 0
        self.time = 1
        self.skip = softstartskip

    def predict(self, Z):
        a = 1
        b = 0
        c = 1

        beta = 0.00001
        alpha = 100

        # 预测步
        x_ = a * self.x_last + self.q  # 预测当前状态
        p_ = a * self.p_last * a + self.Q
        e = Z - x_ - self.r  # 计算残差

        # 更新步
        k = p_ * c / (c * p_ * c + self.R)
        x = x_ + k * e
        p = (1 - k * c) * p_

        # sage husa
        d = -beta * np.log(self.time / alpha) / self.time  # log = ln
        if d > 0:
            self.r = (1 - d) * self.r + d * (Z - x_)
            self.q = (1 - d) * self.q + d * (x - self.x_last)
            self.R = (1 - d) * self.R + d * (e * e - p_)
            self.Q = (1 - d) * self.Q + d * (k * e * e * k + p - self.p_last)

        self.p_last = p
        self.x_last = x
        self.time += 1

        if self.time - 1 > self.skip:
            return x
        else:
            return Z
