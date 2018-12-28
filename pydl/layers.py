"""
Define and configure the layers used in a neural network.
Each layer passes the inputs forward and propagates the gradient backwards.
"""
from typing import Dict
import numpy as np

from pydl.tensor import Tensor

class Layer: 
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads:  Dict[str, Tensor] = {}
    
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Compute the forward pass for the layer.
        """
        raise NotImplementedError
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        Compute gradient and backpropagate it.
        """
        raise NotImplementedError

class Linear(Layer):
    """
    Compute output = input @ W + b
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs will be (batch_size, input_size) 
        # output will be (batch_size, output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)
    
    def forward(self, inputs: Tensor) -> Tensor:
        """
        output = input @ W + b
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and  dy/db = f'(x) * a
        and  dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        and  dy/db = a.T @ f'(x)
        and  dy/dc = f'(x)
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T