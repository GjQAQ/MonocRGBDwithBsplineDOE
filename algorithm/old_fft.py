import torch


class ComplexTensor(torch.Tensor):
    class _MethodWrapper:
        def __init__(self, methods):
            self.__m = methods

        def __call__(self, *args, **kwargs):
            for m in self.__m:
                m(*args, **kwargs)

    def __init__(self, x):
        super().__init__()
        if isinstance(x, tuple):
            self.__real = x[0]
            self.__imag = x[1]
        else:
            self.__real = x[..., 0]
            self.__imag = x[..., 1]

    def __getattr__(self, item):
        origin = getattr(self.__real, item)
        if callable(origin):
            return ComplexTensor._MethodWrapper([
                getattr(self.__real, item),
                getattr(self.__imag, item)
            ])
        else:
            return origin

    def __mul__(self, other):
        return ComplexTensor((
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        ))

    @property
    def real(self):
        return self.__real

    @property
    def imag(self):
        return self.__imag
