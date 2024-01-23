from typing import Optional

import numpy as np
from numpy.typing import NDArray

from distributed_ml.models import LayerInterface


class LinearLayer(LayerInterface):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = np.random.rand(input_dim, output_dim) / output_dim
        self.biases = np.random.randn(output_dim) / output_dim

        self.last_input: Optional[NDArray[np.float32]] = None
        self.weights_grad = np.zeros((input_dim, output_dim))
        self.biases_grad = np.zeros((output_dim,))

    def forward(self, linput: NDArray[np.float32]) -> NDArray[np.float32]:
        assert len(linput.shape) == 2
        assert linput.shape[1] == self.input_dim
        self.last_input = linput
        out = self.last_input @ self.weights + self.biases
        assert out.shape == (linput.shape[0], self.output_dim)
        return out

    def backward(self, upstream_gradient: NDArray) -> NDArray[np.float32]:
        if len(upstream_gradient.shape) == 1:
            upstream_gradient = upstream_gradient[:, None]

        assert len(self.last_input.shape) == len(upstream_gradient.shape) == 2
        assert self.last_input.shape[0] == upstream_gradient.shape[0]
        assert upstream_gradient.shape[1] == self.output_dim

        self.weights_grad = self.last_input.T @ upstream_gradient
        self.biases_grad = upstream_gradient.sum(axis=0)
        assert self.weights_grad.shape == (self.input_dim, self.output_dim)
        assert self.biases_grad.shape == (self.output_dim,)

        downstream_grad = upstream_gradient @ self.weights.T
        assert self.last_input.shape == downstream_grad.shape
        return downstream_grad

    def apply_gradients(self, learning_rate: float) -> None:
        assert learning_rate > 0
        self.weights -= learning_rate * self.weights_grad
        self.biases -= learning_rate * self.biases_grad


class ReLU(LayerInterface):
    def __init__(self):
        self.last_input: Optional[NDArray[np.float32]] = None

    def forward(self, linput: NDArray[np.float32]) -> NDArray[np.float32]:
        assert len(linput.shape) == 2
        self.last_input = linput
        return np.maximum(linput, np.zeros_like(linput))

    def backward(self, upstream_gradient: NDArray[np.float32]) -> NDArray[np.float32]:
        assert self.last_input is not None
        assert self.last_input.shape == upstream_gradient.shape
        return np.where(
            self.last_input >= 0, upstream_gradient, np.zeros_like(self.last_input)
        )

    def apply_gradients(self, learning_rate: float) -> None:
        pass


class MSE(LayerInterface):
    def __init__(self, labels: NDArray[np.float32]) -> None:
        self.labels = labels
        self.last_input: Optional[NDArray[np.float32]] = None

    def forward(self, linput: NDArray[np.float32]) -> NDArray[np.float32]:
        assert self.labels.shape == linput.shape
        self.last_input = linput
        return np.mean(((self.labels - linput) ** 2), axis=0)

    def backward(self, upstream_gradient: NDArray[np.float32]) -> NDArray[np.float32]:
        # 2/n y^ - y
        assert self.last_input is not None
        downstream_gradient = (
            2.0 / self.last_input.shape[0] * (self.last_input - self.labels)
        )
        assert downstream_gradient.shape == self.last_input.shape
        return downstream_gradient

    def apply_gradients(self, learning_rate: float) -> None:
        pass
