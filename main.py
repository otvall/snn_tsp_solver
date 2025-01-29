from src.DTO.neurons import PossibleNeuronModels
from src.solver import TSPSolver
from src.visualiser import Visualiser

if __name__ == "__main__":
    solver = TSPSolver(
        input_file="recourses/five_d.txt",
        neuron_model=PossibleNeuronModels.CUBA,
        feedback_coefficient=-2,
        temp=0.6,
    )
    solver.solve(time=2500)
    print(solver.solver_model.path_data.min_distance)
    print(solver.solver_model.path_data.max_distance)

    visualiser = Visualiser(data_name="fri26_d")
    for i in range(26):
        visualiser.show_wta_dynamic(i)

