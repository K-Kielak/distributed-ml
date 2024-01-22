import os
from collections import deque
from typing import Tuple

import torch
import torch.nn as tnn
import numpy as np
from numpy.typing import NDArray
import pandas as pd


pd.set_option("display.max_columns", None)
DATA_PATH = "housing.csv"


def main() -> None:
    inputs, outputs = load_preprocessed_data(DATA_PATH)
    device = get_device()
    print(f"Inputs shape: {inputs.shape}")
    print(f"Outputs shape: {outputs.shape}")
    model = HousingNn().to(device)
    print(model)
    train(model, inputs, outputs, device)


def train(
    model: tnn.Module,
    inputs: NDArray[np.float64],
    outputs: NDArray[np.float64],
    device: str,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 100,
    early_stopping_history_length: int = 10,
) -> None:
    inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
    outputs = torch.tensor(outputs, dtype=torch.float32, device=device)
    random_ordering = torch.randperm(len(inputs))
    inputs = inputs[random_ordering]
    outputs = outputs[random_ordering]
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping_history = deque()  # for early stopping

    for epoch in range(max_epochs):
        for batch in range(0, len(inputs), batch_size):
            batch_inputs = inputs[batch : batch + batch_size]
            batch_outputs = outputs[batch : batch + batch_size]
            batch_predictions = model(batch_inputs)
            loss = ((batch_predictions - batch_outputs) ** 2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        preds = model(inputs)
        mae = (torch.abs(preds - outputs)).mean()
        mse = ((preds - outputs) ** 2).mean()
        print(
            f"Epoch {epoch + 1} finished. MAE: {mae.item():.3f}, MSE: {mse.item():.3f}"
        )

        if (
            len(early_stopping_history) == early_stopping_history_length
            and np.mean(early_stopping_history) < mae
        ):
            print(f"Early stopping at epoch {epoch + 1}")
            break

        early_stopping_history.append(mae.item())
        if len(early_stopping_history) > early_stopping_history_length:
            early_stopping_history.popleft()


class HousingNn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = tnn.Sequential(
            tnn.Linear(13, 128),
            tnn.ReLU(),
            tnn.Linear(128, 64),
            tnn.ReLU(),
            tnn.Linear(64, 1),
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


def load_preprocessed_data(
    data_path: os.PathLike,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    data = pd.read_csv(data_path)
    data["total_bedrooms"] = data["total_bedrooms"].fillna(
        data["total_bedrooms"].median()
    )
    one_hot_proximity = pd.get_dummies(
        data["ocean_proximity"], prefix="ocean_proximity"
    ).astype(float)
    data = data.drop("ocean_proximity", axis=1)
    data = data.join(one_hot_proximity)
    print(data.describe())
    outputs = data["median_house_value"]
    inputs = data.drop("median_house_value", axis=1)
    return inputs.to_numpy(), outputs.to_numpy()


if __name__ == "__main__":
    main()
