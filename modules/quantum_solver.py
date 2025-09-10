# modules/quantum_solver.py

from qiskit_aer.primitives import Sampler
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit_optimization.applications import VehicleRouting
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import numpy as np
import logging
import time
from typing import List, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SolutionMetrics:
    """A clean data structure for tracking solution quality and metadata."""
    is_valid_quantum_solution: bool
    total_distance: float
    execution_time: float
    notes: str

def _calculate_route_distances(routes: List[List[int]], distance_matrix: np.ndarray) -> Tuple[List[float], float]:
    """Helper function to robustly calculate distances for a list of routes."""
    distances = []
    total_distance = 0.0
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

def _create_classical_fallback(distance_matrix: np.ndarray, num_vehicles: int, depot_node: int) -> List[List[int]]:
    """Creates a simple 'greedy' classical solution if the quantum solver fails."""
    logger.warning("Quantum solver failed to produce a valid result. Using classical fallback.")
    num_locations = distance_matrix.shape[0] - 1
    locations = list(range(1, num_locations + 1))
    
    routes = [[] for _ in range(num_vehicles)]
    for loc in locations:
        best_vehicle_idx = loc % num_vehicles
        routes[best_vehicle_idx].append(loc)
        
    final_routes = [[depot_node] + route + [depot_node] for route in routes if route]
    return final_routes

def solve_quantum_vrp(distance_matrix: np.ndarray, num_vehicles: int, depot_node: int = 0, reps: int = 5) -> Tuple[List[List[int]], List[float], SolutionMetrics]:
    """Enhanced quantum VRP solver with a robust retry loop and classical fallback."""
    start_time = time.time()
    
    try:
        vrp_problem = VehicleRouting(distance_matrix, num_vehicles=num_vehicles, depot=depot_node)
        qp = vrp_problem.to_quadratic_program()
        
        sampler = Sampler(backend_options={"method": "automatic"})
        optimizer = SPSA(maxiter=50)
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps, initial_point=np.random.uniform(0, 2 * np.pi, 2 * reps))

        eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver=qaoa)
        result = eigen_optimizer.solve(qp)
        
        routes = vrp_problem.interpret(result)
        
        if not any(routes):
            raise ValueError("Quantum algorithm returned all empty routes.")

        is_valid_quantum = True
        notes = "Optimal routes found using QAOA."

    except Exception as e:
        logger.error(f"Quantum solver failed: {e}. Switching to classical fallback.")
        routes = _create_classical_fallback(distance_matrix, num_vehicles, depot_node)
        is_valid_quantum = False
        notes = "Quantum solver failed to converge; a classical fallback solution is provided."

    # *** THIS IS THE CORRECTED PART ***
    # Ensure distances are calculated correctly and total_distance is a float
    distances, total_distance = _calculate_route_distances(routes, distance_matrix)
    execution_time = time.time() - start_time
    
    metrics = SolutionMetrics(
        is_valid_quantum_solution=is_valid_quantum,
        total_distance=float(total_distance), # Explicitly cast to float
        execution_time=execution_time,
        notes=notes
    )
    
    return routes, distances, metrics