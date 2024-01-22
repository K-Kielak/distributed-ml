from collections import deque
from typing import Optional

import torch
import torch.nn as tnn
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from distributed_ml.preprocessing import load_preprocessed_data

pd.set_option("display.max_columns", None)
DATA_PATH = "housing.csv"


def main() -> None:
    inputs, outputs, out_mean, out_std = load_preprocessed_data(DATA_PATH)
    device = get_device()
    print(f"Inputs shape: {inputs.shape}")
    print(f"Outputs shape: {outputs.shape}")
    model = HousingNn().to(device)
    print(model)
    train(model, inputs, outputs, device, out_mean=out_mean, out_std=out_std)


def train(
    model: tnn.Module,
    inputs: NDArray[np.float64],
    outputs: NDArray[np.float64],
    device: str,
    learning_rate: float = 1e-4,
    batch_size: int = 512,
    max_epochs: int = 100,
    early_stopping_history_length: int = 10,
    out_mean: Optional[float] = None,
    out_std: Optional[float] = None,
) -> None:
    inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
    outputs = torch.tensor(outputs, dtype=torch.float32, device=device)
    random_ordering = torch.randperm(len(inputs))
    inputs = inputs[random_ordering]
    outputs = outputs[random_ordering]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    early_stopping_history = deque()  # for early stopping

    for epoch in range(max_epochs):
        report_outputs = outputs
        report_preds = model(inputs).squeeze()
        report_mse = ((report_preds - outputs) ** 2).mean()
        mae_suffix = ""
        if out_mean is not None and out_std is not None:
            report_preds = report_preds * out_std + out_mean
            report_outputs = report_outputs * out_std + out_mean
            mae_suffix = " (post destandarization)"

        report_mae = (torch.abs(report_preds - report_outputs)).mean()
        print(
            f"Epoch {epoch + 1} will start. Current MAE{mae_suffix}: {report_mae.item():.3f}, MSE: {report_mse.item():.3f}"
        )

        for batch in range(0, len(inputs), batch_size):
            batch_inputs = inputs[batch : batch + batch_size]
            batch_outputs = outputs[batch : batch + batch_size]
            batch_predictions = model(batch_inputs).squeeze()
            assert batch_predictions.shape == batch_outputs.shape

            loss = ((batch_predictions - batch_outputs) ** 2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (
            len(early_stopping_history) == early_stopping_history_length
            and np.mean(early_stopping_history) < report_mae
        ):
            print(f"Early stopping at epoch {epoch + 1}")
            break

        early_stopping_history.append(report_mae.item())
        if len(early_stopping_history) > early_stopping_history_length:
            early_stopping_history.popleft()


class HousingNn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = tnn.Sequential(
            tnn.Linear(13, 128),
            tnn.ReLU(),
            tnn.Linear(128, 512),
            tnn.ReLU(),
            tnn.Linear(512, 128),
            tnn.ReLU(),
            tnn.Linear(128, 1),
        )

    def forward(self, x):
        return self.layers(x)


def get_device() -> str:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device


if __name__ == "__main__":
    main()
