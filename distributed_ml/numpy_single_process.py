import numpy as np
from numpy.typing import NDArray


def main() -> None:
    smoke_test_linear_layer()


class LinearLayer:
    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = np.random.rand(input_dim, output_dim)
        self.biases = np.random.randn(output_dim)

    def forward(self, linput: NDArray[np.float32]) -> NDArray[np.float32]:
        assert len(linput.shape) == 2
        assert linput.shape[1] == self.input_dim
        out = linput @ self.weights + self.biases
        assert out.shape == (linput.shape[0], self.output_dim)
        return out

    def backward(
        self, linput: NDArray[np.float32], upstream_gradient: NDArray
    ) -> (NDArray[np.float32], NDArray[np.float32]):
        assert len(linput.shape) == len(upstream_gradient.shape) == 2
        assert linput.shape[1] == self.input_dim
        assert linput.shape[0] == upstream_gradient.shape[0]
        assert upstream_gradient.shape[1] == self.output_dim
        weights_gradient = linput.T @ upstream_gradient
        biases_gradient = upstream_gradient.sum(axis=0)
        assert weights_gradient.shape == (self.input_dim, self.output_dim)
        assert biases_gradient.shape == (self.output_dim,)
        return weights_gradient, biases_gradient


def smoke_test_linear_layer():
    batch_size = 50
    input_size = 10
    output_size = 2
    inputs = np.random.randn(batch_size, input_size)
    ll = LinearLayer(input_size, output_size)
    ll.forward(inputs)
    upstream_gradient = np.random.rand(batch_size, output_size)
    ll.backward(inputs, upstream_gradient)
    print("Smoke test for linear layer PASSED!")


if __name__ == "__main__":
    main()
