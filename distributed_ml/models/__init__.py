from abc import abstractmethod, ABC

import numpy as np
from numpy.typing import NDArray


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
