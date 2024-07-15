import os
from tinygrad import Tensor

os.environ["DEBUG"] = "5"
os.environ["NOOPT"] = "1"
os.environ["CUDA"] = "1"
os.environ["GRAPH"] = "1"
os.environ["GRAPHUOPS"] = "1"

a = Tensor([1,2])
b = Tensor([3,4])
res = a.dot(b)
print(res.numpy()) # 11