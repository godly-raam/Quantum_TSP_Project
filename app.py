# app.py

import streamlit as st
import numpy as np
from modules import quantum_solver 
from modules import map_plotter

# --- UI Layout ---
st.set_page_config(layout="wide")
st.title("Quantum Fleet Optimizer ⚛️")
st.write("Optimizing multi-vehicle delivery routes with Quantum Computing (QAOA).")

# --- 1. Initialize Session State ---
if 'routes' not in st.session_state:
    st.session_state.routes = None

# Sidebar for user inputs
with st.sidebar:
    st.header("Problem Settings")
    num_locations = st.slider("Number of Delivery Locations:", min_value=3, max_value=6, value=4)
    num_vehicles = st.slider("Number of Vehicles:", min_value=1, max_value=3, value=2)
    
    st.header("Quantum Algorithm Tuner")
    reps = st.slider("QAOA Quality (reps):", min_value=1, max_value=10, value=5,
                     help="Higher values give better results but take longer to run.")

    st.header("Business Metrics")
    cost_per_km = st.number_input("Cost per Unit Distance ($):", 0.01, 10.0, 2.50)

    seed = st.number_input("Random Seed:", value=123)

# Main button to trigger the optimization
if st.button("Optimize Fleet Routes"):
    
    # Generate the Problem
    np.random.seed(seed)
    depot_node = 0
    coords = np.random.randn(num_locations + 1, 2) * 0.1 + [16.5, 80.5]
    
    distance_matrix = np.zeros((num_locations + 1, num_locations + 1))
    for i in range(num_locations + 1):
        for j in range(i + 1, num_locations + 1):
            dist = np.linalg.norm(coords[i] - coords[j])
            distance_matrix[i, j] = distance_matrix[j, i] = dist
            
    # Solve with Quantum Algorithm
    with st.spinner(f'Running QAOA with reps={reps}... This may take a moment.'):
        routes, distances = quantum_solver.solve_quantum_vrp(
            distance_matrix, 
            num_vehicles, 
            depot_node,
            reps=reps # Pass the new quality parameter
        )
    
    # Save results to session state
    st.session_state.routes = routes
    st.session_state.distances = distances
    st.session_state.coords = coords
    st.session_state.depot_node = depot_node
    st.session_state.cost_per_km = cost_per_km
    st.success("Optimization Complete!")

# Display the results if they exist in the session state
if st.session_state.routes is not None:
    # --- Result Validation ---
    if not any(st.session_state.routes):
        st.warning("Quantum algorithm failed to find a valid solution. Try increasing the QAOA Quality (reps) or changing the seed.")
    else:
        st.subheader("Optimized Routes & Cost")
        total_distance = sum(st.session_state.distances)
        total_cost = total_distance * st.session_state.cost_per_km

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Fleet Distance", f"{total_distance:.2f}")
        with col2:
            st.metric("Estimated Total Cost", f"${total_cost:.2f}")
        
        for i, route in enumerate(st.session_state.routes):
            if route: # Only display non-empty routes
                st.write(f"**Vehicle {i+1} Route:** `{route}` | **Distance:** `{st.session_state.distances[i]:.2f}`")
            
        st.subheader("Route Visualization")
        map_plotter.plot_on_map(
            st.session_state.coords, 
            st.session_state.routes, 
            st.session_state.depot_node
        )
