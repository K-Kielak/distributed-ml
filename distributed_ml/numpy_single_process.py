import os
from abc import ABC, abstractmethod
from typing import Optional

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from numpy.typing import NDArray

from distributed_ml.preprocessing import load_preprocessed_data


DATA_PATH = "housing.csv"


def main() -> None:
    inputs, outputs, out_mean, out_std = load_preprocessed_data(DATA_PATH)
    print(f"Inputs shape: {inputs.shape}")
    print(f"Outputs shape: {outputs.shape}")

    layers = [
        LinearLayer(inputs.shape[1], 128),
        ReLU(),
        LinearLayer(128, 512),
        ReLU(),
        LinearLayer(512, 128),
        ReLU(),
        LinearLayer(128, 1),
    ]
    train(layers, inputs, outputs, out_mean=out_mean, out_std=out_std)


class LayerInterface(ABC):
    @abstractmethod
    def forward(self, linput: NDArray[np.float32]) -> NDArray[np.float32]:
        pass

    @abstractmethod
    def backward(self, upstream_gradient: NDArray[np.float32]) -> NDArray[np.float32]:
        pass

    @abstractmethod
    def apply_gradients(self, learning_rate: float) -> None:
        pass


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


def train(
    layers: list[LayerInterface],
    inputs: NDArray[np.float64],
    outputs: NDArray[np.float64],
    learning_rate: float = 4e-3,
    batch_size: int = 512,
    max_epochs: int = 500,
    out_mean: Optional[float] = None,
    out_std: Optional[float] = None,
) -> None:
    random_ordering = np.arange(inputs.shape[0])
    np.random.shuffle(random_ordering)
    inputs = inputs[random_ordering]
    outputs = outputs[random_ordering]
    for epoch in range(max_epochs):
        report_outputs = outputs
        report_preds = forward_layers(layers, inputs).squeeze()
        report_mse = np.mean((report_preds - report_outputs) ** 2)
        mae_suffix = ""
        if out_mean is not None and out_std is not None:
            report_preds = report_preds * out_std + out_mean
            report_outputs = report_outputs * out_std + out_mean
            mae_suffix = " (post destandarization)"

        report_mae = np.mean(np.abs(report_preds - report_outputs))
        print(
            f"Epoch {epoch + 1} will start. Current MAE{mae_suffix}: {report_mae:.3f}, MSE: {report_mse:.3f}"
        )
        # print(layers[0].weights)

        for batch in range(0, len(inputs), batch_size):
            batch_inputs = inputs[batch : batch + batch_size]
            batch_outputs = outputs[batch : batch + batch_size]
            batch_predictions = forward_layers(layers, batch_inputs).squeeze()

            loss_layer = MSE(batch_outputs)
            loss_layer.forward(batch_predictions)
            loss_gradient = loss_layer.backward(np.empty((0,)))
            backward_layers(layers, loss_gradient, learning_rate)


def forward_layers(
    layers: list[LayerInterface], linput: NDArray[np.float32]
) -> NDArray[np.float32]:
    curr = linput
    for l in layers:
        curr = l.forward(curr)

    return curr


def backward_layers(
    layers: list[LayerInterface],
    loss_gradient: NDArray[np.float32],
    learning_rate: float,
) -> None:
    curr = loss_gradient
    for i, l in enumerate(layers[::-1]):
        curr = l.backward(curr)
        l.apply_gradients(learning_rate)


if __name__ == "__main__":
    main()
