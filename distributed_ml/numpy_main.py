import os
from typing import Optional

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from numpy.typing import NDArray

from distributed_ml.preprocessing import load_preprocessed_data
from distributed_ml.models import LayerInterface
from distributed_ml.models.single_process import LinearLayer, ReLU, MSE
from distributed_ml.models.tensor_parallelism import ParallelLinearLayer


DATA_PATH = "housing.csv"


def main() -> None:
    inputs, outputs, out_mean, out_std = load_preprocessed_data(DATA_PATH)
    print(f"Inputs shape: {inputs.shape}")
    print(f"Outputs shape: {outputs.shape}")

    layers = [
        ParallelLinearLayer(inputs.shape[1], 128),
        ReLU(),
        ParallelLinearLayer(128, 512),
        ReLU(),
        ParallelLinearLayer(512, 128),
        ReLU(),
        ParallelLinearLayer(128, 1),
    ]
    train(layers, inputs, outputs, out_mean=out_mean, out_std=out_std)


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
