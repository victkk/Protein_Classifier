from mynptorch.core.tensor import Tensor
import numpy as np

w = Tensor(np.array([[0, 0, 0]]), requires_grad=True)
b = Tensor(np.array([0]), requires_grad=True)
x = Tensor(np.array([[1, 0, 0], [0, 1, 0]]).T, requires_grad=True)


y_predict = w @ x + b
y_predict.backward()

print(w.grad)
print(b.grad)
print(x.grad)
