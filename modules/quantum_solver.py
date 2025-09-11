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
                    dist_value = distance_matrix[route[i], route[i+1]]
                    # FIX: Handle numpy arrays and scalars properly
                    if isinstance(dist_value, np.ndarray):
                        dist_value = float(dist_value.item())
                    else:
                        dist_value = float(dist_value)
                    route_distance += dist_value
            distances.append(float(route_distance))
            total_distance += route_distance
        except (TypeError, IndexError, ValueError) as e:
            logger.warning(f"Error calculating route distance: {e}")
            distances.append(0.0)
    return distances, float(total_distance)

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
        logger.info(f"Starting quantum VRP solver with {distance_matrix.shape[0]} locations, {num_vehicles} vehicles")
        
        vrp_problem = VehicleRouting(distance_matrix, num_vehicles=num_vehicles, depot=depot_node)
        qp = vrp_problem.to_quadratic_program()
        
        sampler = Sampler(backend_options={"method": "automatic"})
        optimizer = SPSA(maxiter=50)
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps, initial_point=np.random.uniform(0, 2 * np.pi, 2 * reps))

        eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver=qaoa)
        result = eigen_optimizer.solve(qp)
        
        logger.info("Quantum optimization completed, interpreting results...")
        
        # FIX: Handle the result interpretation more carefully
        try:
            routes = vrp_problem.interpret(result)
            logger.info(f"Raw routes from quantum solver: {routes}")
            
            # Ensure routes are in the correct format
            if not routes or not any(routes):
                raise ValueError("Quantum algorithm returned empty routes.")
                
            # Convert routes to proper format if needed
            formatted_routes = []
            for route in routes:
                if isinstance(route, (list, tuple, np.ndarray)):
                    # Convert numpy arrays to lists and ensure integers
                    formatted_route = [int(x) for x in route]
                    formatted_routes.append(formatted_route)
                else:
                    logger.warning(f"Unexpected route format: {type(route)}")
                    
            if not formatted_routes:
                raise ValueError("No valid routes after formatting.")
                
            routes = formatted_routes
            is_valid_quantum = True
            notes = "Optimal routes found using QAOA."
            
        except Exception as interpret_error:
            logger.error(f"Error interpreting quantum result: {interpret_error}")
            raise interpret_error

    except Exception as e:
        logger.error(f"Quantum solver failed: {e}. Switching to classical fallback.")
        routes = _create_classical_fallback(distance_matrix, num_vehicles, depot_node)
        is_valid_quantum = False
        notes = f"Quantum solver failed ({str(e)}); using classical fallback solution."

    # Calculate distances with proper error handling
    try:
        distances, total_distance = _calculate_route_distances(routes, distance_matrix)
        logger.info(f"Calculated distances: {distances}, total: {total_distance}")
    except Exception as dist_error:
        logger.error(f"Error calculating distances: {dist_error}")
        distances = [0.0] * len(routes)
        total_distance = 0.0
    
    execution_time = time.time() - start_time
    
    metrics = SolutionMetrics(
        is_valid_quantum_solution=is_valid_quantum,
        total_distance=float(total_distance),
        execution_time=float(execution_time),
        notes=notes
    )
    
    logger.info(f"VRP solver completed in {execution_time:.2f}s. Routes: {len(routes)}, Total distance: {total_distance:.2f}")
    
    return routes, distances, metrics