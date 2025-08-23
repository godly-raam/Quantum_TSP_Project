# modules/plotter.py

import matplotlib.pyplot as plt
import networkx as nx

def plot_results(coords, classical_route, classical_distance, quantum_route, quantum_distance):
    """
    Visualizes the classical and quantum routes on a 2D plot.
    """
    n = len(coords)
    pos = {i: (coords[i, 0], coords[i, 1]) for i in range(n)}
    
    G = nx.Graph()
    G.add_nodes_from(pos.keys())

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
    nx.draw_networkx_labels(G, pos)

    # Create edge lists from the routes
    classical_edges = list(zip(classical_route[:-1], classical_route[1:]))
    quantum_edges = list(zip(quantum_route[:-1], quantum_route[1:]))

    # Draw the routes on the graph
    nx.draw_networkx_edges(G, pos, edgelist=classical_edges, edge_color='blue', width=2.0, label="Classical")
    nx.draw_networkx_edges(G, pos, edgelist=quantum_edges, edge_color='red', width=2.0, style='dashed', connectionstyle='arc3,rad=0.1', label="Quantum")

    # Add a title and legend
    plt.title(f"TSP Solution Comparison\nClassical: {classical_distance:.2f} | Quantum: {quantum_distance:.2f}")
    plt.legend()
    plt.show()
    
    return plt
