import sys

sys.path.append(r"D:\work\study\ai\PJ\Protein_Classifier\MyNptorch")
from mynptorch.core.tensor import Tensor
import numpy as np

a = Tensor([1, 2, 3, 4, 5], requires_grad=True)
print(a.id)
b = Tensor.clamp(a, min=2, max=4)
print(a.id)
b.backward()
print(a.grad)
