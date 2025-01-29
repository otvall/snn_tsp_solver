import numpy as np
from numpy import ndarray
from torch import Tensor


class PathData:
    def __init__(self, numb_of_cities: int, path_matrix: ndarray) -> None:
        self.path = np.array([-1] * numb_of_cities)
        self.distance = -1
        self.max_distance = -1
        self.min_distance = float("inf")
        self.path_matrix = path_matrix

    def update_distance(self):
        self.distance = self.current_distance()
        self.max_distance = max(self.max_distance, self.distance)
        self.min_distance = min(
            self.min_distance, self.distance if self.distance != -1 else float("inf")
        )

    def current_distance(self) -> int:
        total_length = 0
        if len(self.path) != len(np.unique(self.path)):
            return -1
        if np.count_nonzero(self.path == -1) > 0:
            return -1
        for i in range(-1, len(self.path) - 1):
            city_from = self.path[i]
            city_to = self.path[i + 1]
            total_length += self.path_matrix[city_from][city_to]
        return total_length

    def current_travel_node(self, step: int, block_inference: Tensor) -> None:
        activ_neurons = block_inference[:, :, -1].nonzero()
        self.path[step] = activ_neurons[0, 1].item() if len(activ_neurons) == 1 else -1
