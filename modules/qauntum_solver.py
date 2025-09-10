# modules/quantum_solver.py

from qiskit_aer.primitives import Sampler
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import SPSA
from qiskit_optimization.applications import VehicleRouting
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import numpy as np

def solve_quantum_vrp(distance_matrix, num_vehicles, depot_node=0, reps=5):
    """
    Solves the Vehicle Routing Problem using QAOA.
    """
    vrp_problem = VehicleRouting(
        distance_matrix,
        num_vehicles=num_vehicles,
        depot=depot_node
    )
    qp = vrp_problem.to_quadratic_program()
    
    sampler = Sampler(backend_options={"method": "automatic"})
    optimizer = SPSA(maxiter=50)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps, initial_point=np.zeros(2 * reps))

    optimizer = MinimumEigenOptimizer(min_eigen_solver=qaoa)
    result = optimizer.solve(qp)
    
    routes = vrp_problem.interpret(result)
    
    distances = []
    for route in routes:
        try:
            route_distance = 0
            for i in range(len(route) - 1):
                route_distance += distance_matrix[route[i], route[i+1]]
        except (TypeError, IndexError, ValueError):
            route_distance = 0.0
        
        distances.append(float(route_distance))

    print(f"Solved for {num_vehicles} vehicles with QAOA reps={reps}.")
    return routes, distances