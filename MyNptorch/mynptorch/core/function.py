class Function:
    """可微操作的基类（类似 PyTorch 的 torch.autograd.Function）"""

    def __init__(self):
        self.saved_for_backward = []  # 保存前向传播的输入/输出用于反向传播

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def save_for_backward(self, *tensors, require_grad=False):
        """仅在 requires_grad 为 True 时保存输入"""
        self.saved_for_backward = tensors
