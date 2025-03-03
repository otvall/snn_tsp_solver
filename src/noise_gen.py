import numpy as np
import torch
from torch import Size

from src.DTO.path_data import PathData


class NoiseGenerator:
    def __init__(
        self,
        inference_shape: Size,
        temperature: float,
    ) -> None:
        self.inference_shape = inference_shape
        self.temp = temperature
        self.time = None

    def time_noise(self, t: int) -> torch.Tensor:
        return np.exp(-t / self.time) * (torch.rand(self.inference_shape) * 2 - 1)

    @staticmethod
    def path_noise(distance: int, max_distance: int) -> float:
        if distance != -1 and max_distance != -1:
            return distance / max_distance if max_distance != -1 else 1
        else:
            return 1.

    def gen_noise(self, path_data: PathData, time: int) -> torch.Tensor:
        return (
            self.temp
            * self.path_noise(path_data.distance, path_data.max_distance)
            * self.time_noise(time)
        )
