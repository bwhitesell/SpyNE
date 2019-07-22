import numpy as np


class TensorBase:
    """
    Subclassing np.ndarray is hard and not very beneficial for the use case,
    instead lets just wrap an instance so it is clear which ndarrays are variables
    and which are constants.
    """
    node_uid = ''

    def __init__(self, md_array, name=None):
        if type(md_array) == list:
            self.value = np.array(md_array)
        elif type(md_array) == np.ndarray:
            self.value = md_array
        else:
            raise ValueError("Arguments must be of type 'list' or 'np.ndarray'")

        self._assign_uid()

        if name:
            if type(name) == str:
                self.node_uid = name
            else:
                raise ValueError("Argument 'name' must be of type 'str'")

    def __repr__(self):
        return 'Tensor( \n' + self.value.__str__() + '\n)'

    def _assign_uid(self):
        self.node_uid = str(hash(np.random.random()))

    @property
    def shape(self):
        return self.value.shape


class Tensor(TensorBase):
    name = 'Tensor'


class Constant(TensorBase):
    name = 'Tensor Constant'




