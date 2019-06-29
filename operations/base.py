import numpy as np

from operations.utils import is_ndarray, is_scalar


class BaseOperation:
    name = 'Base Operation'

    def execute(self):
        pass


class UniTensorOperation(BaseOperation):

    def __init__(self, a, b):
        self.a = a
        self._check_types()
        self.value = self.execute()

    def is_single_dim(self):
        return len(self.a.shape) == 1

    def _check_types(self):
        if is_ndarray(self.a):
            if not is_scalar(self.a):
                return True

        raise ValueError(f"""Invalid type, {self.name} is an operation 
        on a tensors of at least dimension 1""")


class DualTensorOperation(BaseOperation):

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self._check_types()
        self.value = self.execute()

    def is_single_dim(self):
        return len(self.a.shape) == 1, len(self.b.shape) == 1

    def _check_types(self):
        if is_ndarray(self.a) and is_ndarray(self.b):
            if not is_scalar(self.a) or not is_scalar(self.b):
                return True

        raise ValueError(f"""Invalid types, {self.name} is an operation 
        between two tensors of at least dimension 1""")
