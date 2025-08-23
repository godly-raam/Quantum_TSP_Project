# modules/classical_solver.py

import networkx as nx

def solve_classical_tsp(distance_matrix):
    """
    Solves the TSP using a classical approximation algorithm from NetworkX.
    """
    G = nx.from_numpy_array(distance_matrix)
    
    # Use NetworkX's built-in TSP solver
    classical_route = nx.approximation.traveling_salesman_problem(G, cycle=True)
    
    # Calculate the total distance
    classical_distance = 0
    for i in range(len(classical_route) - 1):
        classical_distance += G[classical_route[i]][classical_route[i+1]]['weight']
        
    print("--- Classical Solution ---")
    print(f"Route: {classical_route}")
    print(f"Distance: {classical_distance:.2f}")
    
    return classical_route, classical_distance
