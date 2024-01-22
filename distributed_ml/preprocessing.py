import os
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import pandas as pd


def load_preprocessed_data(
    data_path: os.PathLike,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    data = pd.read_csv(data_path)

    # Fill NaNs
    data["total_bedrooms"] = data["total_bedrooms"].fillna(
        data["total_bedrooms"].median()
    )

    # One hot encode
    one_hot_proximity = pd.get_dummies(
        data["ocean_proximity"], prefix="ocean_proximity"
    ).astype(float)
    data = data.drop("ocean_proximity", axis=1)

    # Standardize
    mean = data.mean()
    std = data.std()
    data = (data - mean) / std

    data = data.join(one_hot_proximity)
    print(data.describe())

    # Split
    outputs = data["median_house_value"]
    inputs = data.drop("median_house_value", axis=1)
    return (
        inputs.to_numpy(),
        outputs.to_numpy(),
        mean["median_house_value"],
        std["median_house_value"],
    )
