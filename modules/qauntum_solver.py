# modules/quantum_solver.py

# Imports have been fully corrected for Qiskit 1.0 and later
from qiskit_aer.primitives import Sampler
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import SPSA
from qiskit_optimization.applications import Tsp
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import numpy as np

def solve_quantum_tsp(distance_matrix, seed=123):
    """
    Solves the TSP using the modern Qiskit Primitives (Sampler) with QAOA.
    """
    # 1. Formulate the QUBO
    tsp_problem = Tsp(distance_matrix)
    qp = tsp_problem.to_quadratic_program()
    
    # 2. Set up the modern QAOA algorithm
    sampler = Sampler()
    optimizer = SPSA()
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, initial_point=[0.0, 0.0])

    # 3. Create the main optimization algorithm and solve
    optimizer = MinimumEigenOptimizer(min_eigen_solver=qaoa)
    result = optimizer.solve(qp)
    
    # 4. Interpret and return the results
    # *** THIS IS THE CORRECTED PART ***
    # First, interpret the result to get an integer list for the route
    quantum_route = tsp_problem.interpret(result)
    # Then, calculate the distance using the integer route
    quantum_distance = tsp_problem.tsp_value(quantum_route, distance_matrix)
    
    print("\n--- Quantum Solution ---")
    print(f"Route: {quantum_route}")
    print(f"Distance: {quantum_distance:.2f}")

    return quantum_route, quantum_distance
