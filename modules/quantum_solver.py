# modules/quantum_solver.py - BEST PRACTICE VERSION

from qiskit_aer.primitives import Sampler
from qiskit_aer import AerSimulator
from qiskit_algorithms.minimum_eigensolvers import QAOA, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA, SPSA
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
    is_valid_quantum_solution: bool
    total_distance: float
    execution_time: float
    notes: str

def _calculate_route_distances(routes: List[List[int]], distance_matrix: np.ndarray) -> Tuple[List[float], float]:
    distances = []
    total_distance = 0.0
    for route in routes:
        try:
            route_distance = 0.0
            if len(route) >= 2:
                for i in range(len(route) - 1):
                    dist_value = distance_matrix[route[i], route[i+1]]
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
    logger.warning("Using classical fallback solution.")
    num_locations = distance_matrix.shape[0] - 1
    locations = list(range(1, num_locations + 1))
    
    routes = [[] for _ in range(num_vehicles)]
    for loc in locations:
        best_vehicle_idx = loc % num_vehicles
        routes[best_vehicle_idx].append(loc)
        
    final_routes = [[depot_node] + route + [depot_node] for route in routes if route]
    return final_routes

def solve_quantum_vrp(distance_matrix: np.ndarray, num_vehicles: int, depot_node: int = 0, reps: int = 5) -> Tuple[List[List[int]], List[float], SolutionMetrics]:
    """
    BEST-PRACTICE quantum VRP solver with adaptive method selection.
    Automatically chooses optimal simulation strategy based on problem size.
    """
    start_time = time.time()
    
    # Calculate problem complexity
    num_locations = distance_matrix.shape[0]
    estimated_qubits = (num_locations - 1) * num_vehicles  # Rough estimate
    
    logger.info(f"Problem size: {num_locations} locations, {num_vehicles} vehicles (~{estimated_qubits} qubits)")
    
    try:
        vrp_problem = VehicleRouting(distance_matrix, num_vehicles=num_vehicles, depot=depot_node)
        qp = vrp_problem.to_quadratic_program()
        
        # ============================================
        # ADAPTIVE METHOD SELECTION (THE KEY!)
        # ============================================
        
        if estimated_qubits <= 12:
            # SMALL PROBLEMS: Use exact statevector (best accuracy)
            logger.info("Using EXACT statevector simulation (optimal for small problems)")
            backend = AerSimulator(
                method='statevector',
                device='CPU'
            )
            sampler = Sampler(backend=backend)
            optimizer = COBYLA(maxiter=150)
            adjusted_reps = min(reps, 5)
            shots = None  # Exact computation, no shots needed
            method_note = "exact statevector"
            
        elif estimated_qubits <= 18:
            # MEDIUM PROBLEMS: Use shot-based sampling (memory-efficient)
            logger.info("Using SHOT-BASED sampling (balanced accuracy/memory)")
            backend = AerSimulator(
                method='statevector',
                max_parallel_threads=1,
                max_memory_mb=4096  # 4GB cap
            )
            sampler = Sampler(backend=backend)
            sampler.set_options(shots=2048)  # Good balance
            optimizer = COBYLA(maxiter=100)
            adjusted_reps = min(reps, 4)
            shots = 2048
            method_note = f"sampling ({shots} shots)"
            
        else:
            # LARGE PROBLEMS: Use tensor network approximation or fallback
            logger.warning("Problem too large for standard simulation. Using MPS approximation.")
            backend = AerSimulator(
                method='matrix_product_state',
                matrix_product_state_max_bond_dimension=128,
                max_parallel_threads=1
            )
            sampler = Sampler(backend=backend)
            sampler.set_options(shots=1024)
            optimizer = COBYLA(maxiter=80)
            adjusted_reps = min(reps, 3)
            shots = 1024
            method_note = "tensor network approximation"
        
        # ============================================
        # QAOA EXECUTION
        # ============================================
        
        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=adjusted_reps,
            initial_point=np.random.uniform(0, 2 * np.pi, 2 * adjusted_reps)
        )
        
        eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver=qaoa)
        
        logger.info(f"Running QAOA: {adjusted_reps} layers, method={method_note}")
        result = eigen_optimizer.solve(qp)
        
        # ============================================
        # RESULT INTERPRETATION
        # ============================================
        
        try:
            routes = vrp_problem.interpret(result)
            logger.info(f"Quantum solver returned routes: {routes}")
            
            if not routes or not any(routes):
                raise ValueError("Empty routes returned")
            
            # Format routes
            formatted_routes = []
            for route in routes:
                if isinstance(route, (list, tuple, np.ndarray)):
                    formatted_route = [int(x) for x in route]
                    formatted_routes.append(formatted_route)
            
            if not formatted_routes:
                raise ValueError("No valid routes after formatting")
            
            routes = formatted_routes
            is_valid_quantum = True
            notes = f"QAOA solution ({method_note}, reps={adjusted_reps})"
            
        except Exception as interpret_error:
            logger.error(f"Interpretation failed: {interpret_error}")
            raise interpret_error
    
    except Exception as e:
        logger.error(f"Quantum solver failed: {e}. Using classical fallback.")
        routes = _create_classical_fallback(distance_matrix, num_vehicles, depot_node)
        is_valid_quantum = False
        notes = f"Classical fallback (quantum failed: {str(e)[:50]})"
    
    # Calculate distances
    try:
        distances, total_distance = _calculate_route_distances(routes, distance_matrix)
    except Exception as dist_error:
        logger.error(f"Distance calculation error: {dist_error}")
        distances = [0.0] * len(routes)
        total_distance = 0.0
    
    execution_time = time.time() - start_time
    
    metrics = SolutionMetrics(
        is_valid_quantum_solution=is_valid_quantum,
        total_distance=float(total_distance),
        execution_time=float(execution_time),
        notes=notes
    )
    
    logger.info(f"Solution completed in {execution_time:.2f}s. Total distance: {total_distance:.2f}")
    
    return routes, distances, metrics