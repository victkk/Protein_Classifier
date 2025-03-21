import numpy as np
from ..core.function import Function

from graphviz import Digraph


class Tensor:
    _id_counter = 0  # 类变量用于生成唯一ID

    def __init__(self, data, requires_grad=False, grad_fn=None):
        self.data = np.array(data)
        self.grad = None
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn if requires_grad else None
        self.id = Tensor._id_counter  # 分配唯一ID
        Tensor._id_counter += 1  # 递增计数器

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def _build_topo(self, visited, topo_list):
        """
        Build a topologically sorted list of tensors for backward computation
        """
        if self.id in visited:
            return
        visited.add(self.id)

        # Recursively process all parent tensors
        if self.grad_fn is not None and self.grad_fn.saved_for_backward is not None:
            for tensor in self.grad_fn.saved_for_backward:
                if tensor.requires_grad:
                    tensor._build_topo(visited, topo_list)

        # Add self after all parents have been processed
        topo_list.append(self)

    def backward(self, grad=None):
        """
        Perform backpropagation using topological sort for efficiency
        """
        if not self.requires_grad:
            return  # Skip if gradient computation is not required

        # Initialize gradient for output tensor if not provided
        if grad is None:
            grad = Tensor(np.ones_like(self.data))

        # Initialize gradient for this tensor
        if self.grad is None:
            self.grad = grad.data
        else:
            self.grad += grad.data

        # Build topologically sorted list
        topo_list = []
        visited = set()
        self._build_topo(visited, topo_list)
        # print("tensor in bp num:", len(topo_list))
        # Store grad_values for each tensor to avoid redundant computation
        grad_values = {self.id: grad}

        # Process tensors in reverse topological order (from outputs to inputs)
        for tensor in reversed(topo_list):
            if tensor.id in grad_values:
                current_grad = grad_values[tensor.id]
            else:
                assert 1 == 0
                # This should not happen with correct topo sort, but just in case
                continue

            # Skip if no gradient function
            if tensor.grad_fn is None or tensor.grad_fn.saved_for_backward is None:
                continue

            # Compute gradients for inputs using the operation's backward method
            input_grads = tensor.grad_fn.backward(current_grad)

            # Distribute gradients to parent tensors
            for parent_tensor, parent_grad in zip(
                tensor.grad_fn.saved_for_backward, input_grads
            ):
                if parent_tensor.requires_grad:
                    # Accumulate gradients in the grad_values dictionary
                    if parent_tensor.id in grad_values:
                        grad_values[parent_tensor.id] += parent_grad
                    else:
                        grad_values[parent_tensor.id] = parent_grad

                    # Update the actual tensor's grad attribute
                    if parent_tensor.grad is None:
                        parent_tensor.grad = (
                            parent_grad.data
                            if isinstance(parent_grad, Tensor)
                            else parent_grad
                        )
                    else:
                        parent_tensor.grad += (
                            parent_grad.data
                            if isinstance(parent_grad, Tensor)
                            else parent_grad
                        )

    def visualize_backward(self, filename="compute_graph", format="png", view=True):
        """
        可视化计算图结构

        参数:
            self (Tensor): 需要可视化的起始Tensor
            filename (str): 输出文件名（默认'compute_graph'）
            format (str): 输出格式如'png', 'pdf'等（默认'png'）
            view (bool): 是否自动打开生成的文件
        """
        dot = Digraph(name=filename, format=format)
        visited = set()

        def add_node(tensor_or_fn):
            if id(tensor_or_fn) in visited:
                return
            visited.add(id(tensor_or_fn))

            if isinstance(tensor_or_fn, Tensor):
                # 精简数据表示
                data_str = (
                    str(tensor_or_fn.data).replace("\n", ", ")[:20] + "..."
                    if tensor_or_fn.data.size > 3
                    else str(tensor_or_fn.data)
                )
                grad_str = (
                    str(tensor_or_fn.grad).replace("\n", ", ")[:20] + "..."
                    if tensor_or_fn.grad is not None and tensor_or_fn.grad.size > 3
                    else str(tensor_or_fn.grad)
                )
                label = (
                    f"Tensor [{tensor_or_fn.id}]\n"
                    f"Shape: {tensor_or_fn.data.shape}\n"
                    f"Data: {data_str}\n"
                    f"Grad: {grad_str}\n"
                    f"Requires Grad: {tensor_or_fn.requires_grad}"
                )
                dot.node(str(id(tensor_or_fn)), label, shape="ellipse")

            elif isinstance(tensor_or_fn, Function):
                label = f"Function [{type(tensor_or_fn).__name__}]\n"
                if tensor_or_fn.saved_for_backward is not None:
                    label += f"Saved Tensors: {len(tensor_or_fn.saved_for_backward)}"
                dot.node(
                    str(id(tensor_or_fn)),
                    label,
                    shape="rectangle",
                    style="filled",
                    fillcolor="#e0e0e0",
                )

        # 递归遍历函数
        def traverse(node):
            if isinstance(node, Tensor):
                add_node(node)
                if node.grad_fn is not None:
                    add_node(node.grad_fn)
                    dot.edge(str(id(node.grad_fn)), str(id(node)), label="Output")
                    for input_tensor in node.grad_fn.saved_for_backward:
                        if isinstance(input_tensor, Tensor):
                            add_node(input_tensor)
                            dot.edge(
                                str(id(input_tensor)),
                                str(id(node.grad_fn)),
                                label="Input",
                            )
                            traverse(input_tensor)

        # 开始遍历
        traverse(self)

        # 添加梯度关系（虚线表示）
        def add_grad_edges(node):
            if isinstance(node, Tensor) and node.grad_fn is not None:
                for input_tensor in node.grad_fn.saved_for_backward:
                    if (
                        isinstance(input_tensor, Tensor)
                        and input_tensor.grad is not None
                    ):
                        dot.edge(
                            str(id(node)),
                            str(id(input_tensor)),
                            style="dashed",
                            color="#606060",
                            label="Grad",
                        )
                        add_grad_edges(input_tensor)

        add_grad_edges(self)

        # 渲染并显示
        dot.render(filename=filename, cleanup=True, view=view)

    # ----------------- 运算符重载 -----------------
    def __add__(self, other):
        return add(self, _ensure_tensor(other))

    def __mul__(self, other):
        return mul(self, _ensure_tensor(other))

    def __matmul__(self, other):
        return matmul(self, _ensure_tensor(other))

    def __pow__(self, other):
        return power(self, _ensure_tensor(other))

    def __sub__(self, other):
        return add(self, mul(_ensure_tensor(other), Tensor(-1)))

    def __truediv__(self, other):
        return mul(self, power(_ensure_tensor(other), Tensor(-1)))
        # 新增的log和exp方法

    def log(self):
        return log(self)

    def exp(self):
        return exp(self)

    def mean(self, axis=None, keepdims=False):
        return mean(self, axis=axis, keepdims=keepdims)

    def sum(self, axis=None, keepdims=False):
        return sum(self, axis=axis, keepdims=keepdims)

    # 反向运算符
    # todo check
    __radd__ = __add__
    __rmul__ = __mul__

    def __rmatmul__(self, other):
        return matmul(_ensure_tensor(other), self)

    def __rsub__(self, other):
        return add(_ensure_tensor(other), mul(self, Tensor(-1)))

    def __rtruediv__(self, other):
        return mul(_ensure_tensor(other), power(self, Tensor(-1)))

    def __neg__(self):
        return mul(self, Tensor(-1))


# ----------------- 核心运算实现 -----------------
class Add(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x.data + y.data

    def backward(self, grad_output):
        return grad_output, grad_output  # 加法梯度分发规则


class Mul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x.data * y.data

    def backward(self, grad_output):
        x, y = self.saved_for_backward
        return grad_output * y.data, grad_output * x.data  # 乘法梯度规则


class MatMul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x.data @ y.data

    def backward(self, grad_output):
        x, y = self.saved_for_backward
        return (
            grad_output @ Tensor(y.data.T),
            Tensor(x.data.T) @ grad_output,
        )  # 矩阵乘法梯度


class Power(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x.data**y.data

    def backward(self, grad_output):
        x, y = self.saved_for_backward
        dx = grad_output * y.data * (x.data ** (y.data - 1))
        dy = grad_output * (x.data**y.data) * np.log(x.data)
        return dx, dy


# ----------------- 工具函数 -----------------
def _ensure_tensor(other):
    """将标量或numpy数组转换为Tensor"""
    if not isinstance(other, Tensor):
        return Tensor(other, requires_grad=False)
    return other


def add(x, y):
    f = Add()
    out_data = f.forward(x, y)
    requires_grad = x.requires_grad or y.requires_grad
    return Tensor(out_data, requires_grad=requires_grad, grad_fn=f)


def mul(x, y):
    f = Mul()
    out_data = f.forward(x, y)
    requires_grad = x.requires_grad or y.requires_grad
    return Tensor(out_data, requires_grad=requires_grad, grad_fn=f)


def matmul(x, y):
    f = MatMul()
    out_data = f.forward(x, y)
    requires_grad = x.requires_grad or y.requires_grad
    return Tensor(out_data, requires_grad=requires_grad, grad_fn=f)


def power(x, y):
    f = Power()
    out_data = f.forward(x, y)
    requires_grad = x.requires_grad or y.requires_grad
    return Tensor(out_data, requires_grad=requires_grad, grad_fn=f)
    # 其他运算符重载保持不变
    # ... [原有代码保持不变] ...


# 新增的LogFunction和ExpFunction类
class LogFunction(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return np.log(x.data)

    def backward(self, grad_output):
        x = self.saved_for_backward[0]
        return (grad_output * (1 / x.data),)


class ExpFunction(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return np.exp(x.data)

    def backward(self, grad_output):
        x = self.saved_for_backward[0]
        return (grad_output * np.exp(x.data),)


class SumFunction(Function):
    def forward(self, x, axis=None, keepdims=False):
        self.save_for_backward(x)
        self.axis = axis
        self.keepdims = keepdims
        return np.sum(x.data, axis=axis, keepdims=keepdims)

    def backward(self, grad_output):
        x = self.saved_for_backward[0]
        grad_data = grad_output.data
        if not self.keepdims and self.axis is not None:
            if isinstance(self.axis, int):
                grad_data = np.expand_dims(grad_data, axis=self.axis)
            else:
                for ax in sorted(self.axis):
                    grad_data = np.expand_dims(grad_data, axis=ax)
        grad = np.ones_like(x.data) * grad_data
        return (Tensor(grad),)


class MeanFunction(Function):
    def forward(self, x, axis=None, keepdims=False):
        self.save_for_backward(x)
        self.axis = axis
        self.keepdims = keepdims
        return np.mean(x.data, axis=axis, keepdims=keepdims)

    def backward(self, grad_output):
        x = self.saved_for_backward[0]
        grad_data = grad_output.data

        # 计算梯度缩放因子
        if self.axis is None:
            # 全局平均，除以总元素个数
            scale = 1 / x.data.size
        else:
            # 沿指定轴平均，计算参与平均的元素个数
            if isinstance(self.axis, int):
                size = x.data.shape[self.axis]
            else:
                size = np.prod([x.data.shape[ax] for ax in self.axis])
            scale = 1 / size

        # 处理维度广播
        if not self.keepdims and self.axis is not None:
            # 插入被缩减的维度
            if isinstance(self.axis, int):
                grad_data = np.expand_dims(grad_data, axis=self.axis)
            else:
                for ax in sorted(self.axis):
                    grad_data = np.expand_dims(grad_data, axis=ax)

        # 生成梯度矩阵
        grad = np.ones_like(x.data) * grad_data * scale
        return (Tensor(grad),)


def mean(x, axis=None, keepdims=False):
    f = MeanFunction()
    out_data = f.forward(x, axis=axis, keepdims=keepdims)
    return Tensor(out_data, requires_grad=x.requires_grad, grad_fn=f)


def sum(x, axis=None, keepdims=False):
    f = SumFunction()
    out_data = f.forward(x, axis=axis, keepdims=keepdims)
    return Tensor(out_data, requires_grad=x.requires_grad, grad_fn=f)


# 新增的log和exp工具函数
def log(x):
    f = LogFunction()
    out_data = f.forward(x)
    requires_grad = x.requires_grad
    return Tensor(out_data, requires_grad=requires_grad, grad_fn=f)


def exp(x):
    f = ExpFunction()
    out_data = f.forward(x)
    requires_grad = x.requires_grad
    return Tensor(out_data, requires_grad=requires_grad, grad_fn=f)
