import numpy as np


class TensorBase:
    """
    Subclassing np.ndarray is hard and not very beneficial for the use case,
    instead lets just wrap an instance so it is clear which ndarrays are variables
    and which are constants.
    """

    def __init__(self, md_array):
        if type(md_array) == list:
            self.value = np.array(md_array)
        elif type(md_array) == np.ndarray:
            self.value = md_array
        else:
            raise ValueError("Arguments must be of type 'list' or 'np.ndarray'")

    def __repr__(self):
        return 'Tensor( \n' + self.value.__str__() + '\n)'


class Tensor(TensorBase):
    name = 'Tensor'


class TensorConst:
    name = 'Tensor Constant'




