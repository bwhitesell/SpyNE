from operations.arithmetic import TensorAddition, TensorSubtraction
from variables import Tensor


A = Tensor([[[0.52, 1.12, 2.34],
               [0.88, -1.08, 1.56],
               [0.52, 1.12, 2.34]],

              [[0.88, -1.08, 1.56],
               [0.52, 1.12, 2.34],
               [0.88, -1.08, 1.56]]])

B = Tensor([[[1.52, 4.22, 2.34],
               [-2.42, 6.11, 5.24],
               [1.52, 2.13, 21.56]],

              [[4.31, 1.08, 1.22],
               [2.54, 1.12, 2.54],
               [3.32, -1.32, 1.66]]])

C = TensorAddition(A, B)
D = TensorAddition(C, A)

print(C)




