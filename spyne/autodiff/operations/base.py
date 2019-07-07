import numpy as np

from ..variables.variables import Tensor, TensorConst


class BaseOperation:
    name = 'Base Operation'
    node_uid = ''
    children = {}

    def __init__(self):
        self._assign_uid()
        self.value = self.execute()

    def __setattr__(self, key, value):
        if hasattr(value, 'value'):
            self.__dict__[key] = value
            self.__dict__['_' + key] = value.value
        else:
            self.__dict__[key] = value

    def __repr__(self):
        return 'Tensor( \n' + self.value.__str__() + '\n)'

    def _assign_uid(self):
        self.node_uid = str(hash(np.random.random()))

    def execute(self):
        pass

    @staticmethod
    def is_operation(x):
        return issubclass(x.__class__, BaseOperation)

    @staticmethod
    def is_tensor(x):
        return type(x) == Tensor or type(x) == TensorConst

    @property
    def shape(self):
        return self.value.shape


class UniTensorOperation(BaseOperation):

    def __init__(self, a):
        self._check_type(a)
        self.a = a
        super().__init__()

    def is_single_dim(self):
        return len(self._a.shape) == 1

    def _check_type(self, a):
        if not self.is_tensor(a) and not self.is_operation(a):
            raise ValueError(f"""Invalid type, {self.name} is an operation 
            on a Tensor object of at least dimension 1""")


class DualTensorOperation(BaseOperation):

    def __init__(self, a, b):
        self._check_types(a, b)
        self.a = a
        self.b = b
        super().__init__()

    def is_single_dim(self):
        return len(self.a.shape) == 1, len(self.b.shape) == 1

    def _check_types(self, a, b):
        if (not self.is_tensor(a) and not self.is_operation(a)) or (not self.is_tensor(b)
                                                                    and not self.is_operation(b)):
            raise ValueError(f"""Invalid types, {self.name} is an operation 
                             between two tensors of at least dimension 1""")
