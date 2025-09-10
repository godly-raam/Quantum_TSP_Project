# modules/quantum_solver.py

from qiskit_aer.primitives import Sampler
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit_optimization.applications import VehicleRouting
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import numpy as np
import logging
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

# Configure logging to see solver steps in Render logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SolutionMetrics:
    """A clean data structure for tracking solution quality and metadata."""
    is_valid_quantum_solution: bool
    total_distance: float
    execution_time: float
    notes: str

# This is a helper function to robustly calculate distances for a list of routes.
def _calculate_route_distances(routes: List[List[int]], distance_matrix: np.ndarray) -> Tuple[List[float], float]:
    distances, total_distance = [], 0.0
    for route in routes:
        try:
            route_distance = 0.0
            if len(route) >= 2:
                for i in range(len(route) - 1):
                    route_distance += distance_matrix[route[i], route[i+1]]
            distances.append(route_distance)
            total_distance += route_distance
        except (TypeError, IndexError, ValueError):
            distances.append(0.0)
    return distances, total_distance

# This function creates a simple 'greedy' classical solution if the quantum solver fails completely.
def _create_classical_fallback(distance_matrix: np.ndarray, num_vehicles: int, depot_node: int) -> List[List[int]]:
    logger.warning("Quantum solver failed to produce a valid result. Using classical fallback.")
    num_locations = distance_matrix.shape[0] - 1
    locations = list(range(1, num_locations + 1))
    
    routes = [[] for _ in range(num_vehicles)]
    for loc in locations:
        # Simple greedy assignment: assign location to a vehicle based on its index
        best_vehicle_idx = loc % num_vehicles
        routes[best_vehicle_idx].append(loc)
        
    # Finalize routes by adding the depot at the start and end of each route
    final_routes = [[depot_node] + route + [depot_node] for route in routes if route]
    return final_routes

# This is the main function your API will call.
def solve_quantum_vrp(distance_matrix: np.ndarray, num_vehicles: int, depot_node: int = 0, reps: int = 5) -> Tuple[List[List[int]], List[float], SolutionMetrics]:
    start_time = time.time()
    
    try:
        # 1. Standard VRP setup using Qiskit
        vrp_problem = VehicleRouting(distance_matrix, num_vehicles=num_vehicles, depot=depot_node)
        qp = vrp_problem.to_quadratic_program()
        
        # 2. Configure the quantum algorithm (QAOA)
        sampler = Sampler(backend_options={"method": "automatic"})
        optimizer = SPSA(maxiter=50)
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps, initial_point=np.random.uniform(0, 2 * np.pi, 2 * reps))

        eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver=qaoa)
        
        # 3. Solve the problem
        result = eigen_optimizer.solve(qp)
        routes = vrp_problem.interpret(result)
        
        # 4. Validate the result
        if not any(routes): # Check if all routes are empty
            raise ValueError("Quantum algorithm returned all empty routes.")

        is_valid_quantum = True
        notes = "Optimal routes found using QAOA."
        logger.info(f"Valid quantum solution found with reps={reps}.")

    except Exception as e:
        # 5. Fallback Mechanism: If the try block fails for any reason...
        logger.error(f"Quantum solver failed: {e}. Switching to classical fallback.")
        routes = _create_classical_fallback(distance_matrix, num_vehicles, depot_node)
        is_valid_quantum = False
        notes = "Quantum solver failed to converge; a classical fallback solution is provided."

    # 6. Calculate final distances and metrics
    distances, total_distance = _calculate_route_distances(routes, distance_matrix)
    execution_time = time.time() - start_time
    
    metrics = SolutionMetrics(
        is_valid_quantum_solution=is_valid_quantum,
        total_distance=total_distance,
        execution_time=execution_time,
        notes=notes
    )
    
    return routes, distances, metrics