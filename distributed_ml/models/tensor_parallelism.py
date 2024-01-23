import math
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from distributed_ml.models import LayerInterface


class ParallelLinearLayer(LayerInterface):
    def __init__(
        self, input_dim: int, output_dim: int, n_workers: int = 2, n_tasks: int = 4
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.n_workers = n_workers
        self.n_tasks = n_tasks

        self.weights = np.random.rand(input_dim, output_dim) / self.output_dim
        self.biases = np.random.rand(output_dim) / self.output_dim

        self.weights_grads = np.zeros((input_dim, output_dim))
        self.biases_grads = np.zeros((output_dim,))

        self.last_input: Optional[NDArray[np.float32]] = None

    def forward(self, linput: NDArray[np.float32]) -> NDArray[np.float32]:
        assert linput.shape[1] == self.input_dim
        self.last_input = linput
        chunk_size = math.ceil(self.output_dim / self.n_tasks)
        linput_iter = [linput] * self.n_tasks
        weights_iter = [
            self.weights[:, t * chunk_size : (t + 1) * chunk_size]
            for t in range(self.n_tasks)
        ]
        biases_iter = [
            self.biases[t * chunk_size : (t + 1) * chunk_size]
            for t in range(self.n_tasks)
        ]
        with ProcessPoolExecutor(self.n_workers) as thread_pool:
            results = thread_pool.map(
                self._single_forward, linput_iter, weights_iter, biases_iter
            )

        return np.concatenate(list(results), axis=1)

    def _single_forward(
        self,
        linput: NDArray[np.float32],
        weights: NDArray[np.float32],
        biases: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        assert linput.shape[1] == weights.shape[0]
        assert weights.shape[1] == biases.shape[0]
        return linput @ weights + biases

    def backward(self, upstream_gradient: NDArray[np.float32]) -> NDArray[np.float32]:
        if len(upstream_gradient.shape) == 1:
            upstream_gradient = upstream_gradient[:, None]

        chunk_size = math.ceil(self.output_dim / self.n_tasks)
        gradient_iter = [
            upstream_gradient[:, t * chunk_size : (t + 1) * chunk_size]
            for t in range(self.n_tasks)
        ]
        weights_iter = [
            self.weights[:, t * chunk_size : (t + 1) * chunk_size]
            for t in range(self.n_tasks)
        ]
        with ProcessPoolExecutor(max_workers=self.n_workers) as thread_pool:
            results = thread_pool.map(
                self._single_backward, gradient_iter, weights_iter
            )

        weights_res, bias_res, downstream_res = zip(*results)
        self.weights_grads = np.concatenate(weights_res, axis=1)
        assert self.weights_grads.shape == (self.input_dim, self.output_dim)
        self.biases_grads = np.concatenate(bias_res, axis=0)
        assert self.biases_grads.shape == (self.output_dim,)
        downstream_grads = sum(downstream_res)
        assert downstream_grads.shape == self.last_input.shape
        return downstream_grads

    def _single_backward(
        self, upstream_gradient: NDArray[np.float32], weights: NDArray[np.float32]
    ) -> (NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]):
        # we want: dL/dW, dL/dB, dL/dX
        # dL/dO * dO/dW
        # O = XW + B
        # dO/dW = X
        # dO/dX = W
        # dO/dB = 1
        # Get some gradient from upstream called G (of shape B x O)
        # dL/dW = (X^T)B - shape I x O (like W so all good)
        # dL/dX = G(W^T) - shape B x I (like X so all good)
        # dL/dB = np.sum(G, axis=0)?
        assert self.last_input is not None
        assert self.last_input.shape[1] == weights.shape[0]
        assert upstream_gradient.shape[0] == self.last_input.shape[0]
        assert upstream_gradient.shape[1] == weights.shape[1]

        # orginal upstream grad shape: b x o
        # chunked ug shape: b x o/tasks
        # original weights shape: i x o
        # chunked weights shape: i x o/tasks
        # last input shape: b x i
        # weights grad shape: i x b @ b x o/task = i x o/tasks
        # bias grad shape: o/tasks
        # downstream grad shape: b x o/tasks @ o/tasks x i = b x i
        weights_grad = self.last_input.T @ upstream_gradient
        bias_grad = np.sum(upstream_gradient, axis=0)
        downstream_grad = upstream_gradient @ weights.T
        return weights_grad, bias_grad, downstream_grad

    def apply_gradients(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.weights_grads
        self.biases -= learning_rate * self.biases_grads
