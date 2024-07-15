import os
from tinygrad import Tensor

# NOOPT=1
os.environ["NOOPT"] = "1"


full = Tensor.full(shape=(2, 3), fill_value=5) # create a tensor of shape (2, 3) filled with 5
zeros = Tensor.zeros(2, 3) # create a tensor of shape (2, 3) filled with 0
ones = Tensor.ones(2, 3) # create a tensor of shape (2, 3) filled with 1

full_like = Tensor.full_like(full, fill_value=2) # create a tensor of the same shape as `full` filled with 2
zeros_like = Tensor.zeros_like(full) # create a tensor of the same shape as `full` filled with 0
ones_like = Tensor.ones_like(full) # create a tensor of the same shape as `full` filled with 1

eye = Tensor.eye(3) # create a 3x3 identity matrix
arange = Tensor.arange(start=0, stop=10, step=1) # create a tensor of shape (10,) filled with values from 0 to 9

rand = Tensor.rand(2, 3) # create a tensor of shape (2, 3) filled with random values from a uniform distribution
randn = Tensor.randn(2, 3) # create a tensor of shape (2, 3) filled with random values from a standard normal distribution
uniform = Tensor.uniform(2, 3, low=0, high=10) # create a tensor of shape (2, 3) filled with random values from a uniform distribution between 0 and 10

print(arange.numpy())


a = Tensor([1,2])
b = Tensor([3,4])
res = a.dot(b)
# print(res.numpy()) # 11