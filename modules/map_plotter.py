# modules/map_plotter.py

import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np

def plot_on_map(coords, routes, depot_node=0):
    """
    Creates and displays an interactive Folium map of the VRP routes.
    """
    # Center the map on the average location
    map_center = np.mean(coords, axis=0)
    m = folium.Map(location=map_center, zoom_start=11)

    # Add markers for each location
    for i, coord in enumerate(coords):
        tooltip = f"Location {i}"
        popup = f"Coordinates: ({coord[0]:.2f}, {coord[1]:.2f})"
        
        if i == depot_node:
            folium.Marker(
                location=coord,
                tooltip="Depot",
                popup=f"Depot {i}",
                icon=folium.Icon(color='red', icon='industry', prefix='fa')
            ).add_to(m)
        else:
            folium.Marker(
                location=coord,
                tooltip=tooltip,
                popup=popup,
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)
            
    # Define colors for the vehicle routes
    colors = ['#FF0000', '#0000FF', '#008000', '#FFA500', '#800080'] # Red, Blue, Green, Orange, Purple
    
    # Add polylines for each route
    for i, route in enumerate(routes):
        if route: # Only plot valid routes
            route_coords = [coords[j] for j in route]
            folium.PolyLine(
                locations=route_coords,
                color=colors[i % len(colors)],
                weight=5,
                opacity=0.8,
                tooltip=f"Vehicle {i+1} Route"
            ).add_to(m)

    # Display the map in the Streamlit app
    st_folium(m, width=725, height=500)
