import torch
from torch import Tensor
from torch.nn import ModuleList
from tqdm import tqdm

from .commutator import BlockCommutator
from .data_monitor import DataMonitor
from .DTO.path_data import PathData
from .noise_gen import NoiseGenerator
from src.DTO.neurons import PossibleNeuronModels


class TSPModel:
    def __init__(
        self,
        numb_of_cities: int,
        feedback_coefficient: float,
        weights: Tensor,
        neuron_model: PossibleNeuronModels,
        path_data: PathData,
        temp: float,
        data_monitor: DataMonitor,
    ) -> None:
        self.numb_of_cities = numb_of_cities
        self.feedback_coefficient = feedback_coefficient
        self.path_data = path_data
        self.data_monitor = data_monitor
        self.start_spike = self._init_start_spike(numb_of_cities)
        self.blocks = self._init_wta_blocks(numb_of_cities, weights, neuron_model)
        self.commutator = self._init_block_commutator(self.start_spike, numb_of_cities)
        self.noise_generator = self._init_noise_generator(self.start_spike, temp)

    @staticmethod
    def _init_block_commutator(
        start_spike: Tensor, numb_of_cities: int
    ) -> BlockCommutator:
        return BlockCommutator(
            inference_shape=start_spike.shape,
            number_of_cities=numb_of_cities,
        )

    @staticmethod
    def _init_noise_generator(
        start_spike: Tensor, temperature: float
    ) -> NoiseGenerator:
        return NoiseGenerator(
            inference_shape=start_spike.shape,
            temperature=temperature,
        )

    @staticmethod
    def _init_wta_blocks(
        numb_of_cities: int, weights: Tensor, neuron_model: PossibleNeuronModels
    ) -> ModuleList:
        block_model = neuron_model.value.block_model
        neurons_config = neuron_model.value.neuron_config
        blocks = ModuleList(
            [
                block_model(
                    neuron_params=neurons_config,
                    in_neurons=numb_of_cities,
                    out_neurons=numb_of_cities,
                    num_winners=1,
                    delay_shift=False,
                    self_excitation=0.2,
                )
                for _ in range(numb_of_cities)
            ]
        )

        custom_weights_param = torch.nn.Parameter(
            weights.reshape(shape=(numb_of_cities, numb_of_cities, 1, 1, 1))
        )

        for block in blocks:
            block.synapse.weight = custom_weights_param

        return blocks

    @staticmethod
    def _init_start_spike(numb_of_cities: int) -> Tensor:
        start_spike = torch.zeros(1, numb_of_cities, 2)
        start_spike[:, 0, :] += 1.0
        return start_spike

    def forward(self, input_spike: Tensor, t: int) -> Tensor:
        for i, block in enumerate(self.blocks):
            block.bias = self.feedback_coefficient * self.commutator.get_bias(port=i)

            input_spike += self.noise_generator.gen_noise(
                path_data=self.path_data,
                time=t,
            )
            input_spike = block(input_spike)
            self.data_monitor.save_wta_inference(input_spike, i)

            self.path_data.current_travel_node(i, input_spike)
            self.commutator.set_bias(port=i, bias=input_spike)

        return input_spike

    def solve(self, time: int) -> None:
        input_spike = self.start_spike.clone()
        self.noise_generator.time = time

        for t in tqdm(range(time)):
            input_spike = self.forward(input_spike, t)
            self.path_data.update_distance()
            self.data_monitor.save_model_inference(self.path_data)
