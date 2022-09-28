class FixedLengthList(list):
    def __init__(self, length):
        super(list).__init__()
        self.mlen = length

    def append(self, *args, **kwargs):
        if self.__len__() > self.mlen-1:
            super().pop(0)
        super().append(args[0])


if __name__ == '__main__':
    # for debug
    li = FixedLengthList(3)
    print()
