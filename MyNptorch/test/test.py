import sys

sys.path.append("../")
from mynptorch.core.tensor import Tensor
import numpy as np
import pandas as pd


def loss(y_pred, y):
    epsilon = 1e-10
    loss = y * Tensor.log(y_pred + epsilon) + (1 - y) * Tensor.log(1 - y_pred + epsilon)
    return -Tensor.mean(loss)


def backward(weight, bias, x, y):
    print(weight.shape, bias.shape, x.shape, y.shape)
    N = x.shape[1]
    z = weight @ x + bias
    y_pred = 1 / (1 + np.exp(-z))
    dl_dz = -1 / N * ((1 - y_pred) * y - y_pred * (1 - y))
    dl_dw = dl_dz @ x.T
    dl_db = np.sum(dl_dz)
    print("weight grad:", dl_dw)
    print("bias grad:", dl_db)
    return dl_dw, dl_db


def data_preprocess():
    diagrams = np.load("./data/diagrams.npy")
    cast = pd.read_table("./data/SCOP40mini_sequence_minidatabase_19.cast")
    cast.columns.values[0] = "protein"
    # standardize the data
    mean = np.mean(diagrams, axis=0)
    std = np.std(diagrams, axis=0)
    diagrams = (diagrams - mean) / (std + 1e-8)

    data_list = []
    target_list = []
    for task in range(1, 56):  # Assuming only one task for now
        task_col = cast.iloc[:, task].to_numpy()

        train_mask = (task_col == 1) | (task_col == 2)
        test_mask = (task_col == 3) | (task_col == 4)
        train_data = diagrams[train_mask]
        test_data = diagrams[test_mask]
        # train_data = train_data[:100]
        train_targets = task_col[train_mask] * (-1) + 2
        # train_targets = train_targets[:100]
        test_targets = task_col[test_mask] * (-1) + 4

        assert train_targets.shape[0] == train_data.shape[0]
        assert test_targets.shape[0] == test_data.shape[0]

        data_list.append((train_data, test_data))
        target_list.append((train_targets, test_targets))

    return data_list, target_list


data_list, target_list = data_preprocess()
x = Tensor(data_list[0][0][:10, :].T, requires_grad=True)
l = Tensor(target_list[0][0][:10], requires_grad=True)
print(x.data.shape, l.data.shape)
print(x.data.dtype, l.data.dtype)
print(l.data)
w = Tensor(
    np.zeros((1, 300)),
    requires_grad=True,
)
b = Tensor(np.zeros(1), requires_grad=True)
# x = Tensor(np.array([[1.21, 5.212, 1], [6.77, 1, 1]]).T, requires_grad=True)
# l = Tensor(np.array([1, 0]), requires_grad=True)
print(x.data.shape, l.data.shape)

print(x.data.dtype, l.data.dtype)
print(l.data)
for i in range(200):
    y_predict1 = w @ x + b
    # y_predict1.id = 0
    y_predict2 = 1 / (1 + Tensor.exp(-y_predict1))
    loss1 = loss(y_predict2, l)
    # y_predict2.backward(grad=Tensor([[-1, 0.4]]))
    loss1.backward()
    loss1.visualize_backward()
    w_grad, b_grad = backward(w.data, b.data, x.data, l.data)
    print("2grad")
    print(w.grad)
    print(w_grad)
    print(w_grad / w.grad.data)

    w -= 1e-1 * w.grad
    b -= 1e-1 * np.mean(b.grad)
    # print(loss1)
    # # print(y_predict1)
    # print(y_predict2)

    print()
loss1.visualize_backward()
print("pred1:", y_predict1.grad)
print("pred2", y_predict2.grad)
print(w.grad, b.grad)
