# main.py

import sys
# This tells Python to also look for files in the 'modules' directory
sys.path.append('./modules')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
# This import will now work correctly
from modules import quantum_solver

app = FastAPI()

# --- IMPORTANT: Add CORS middleware ---
# This allows your Vercel frontend to make requests to your Render backend
origins = [
    "https://entangled-minds-qc.vercel.app",
    "http://localhost:3000", # Add for local frontend testing
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VrpProblem(BaseModel):
    num_locations: int
    num_vehicles: int
    reps: int = 5

@app.post("/api/optimize")
def optimize_routes(problem: VrpProblem):
    np.random.seed(123) # Use a fixed seed for consistent results
    depot_node = 0
    # Use real-world-like coordinates for consistency
    coords = np.random.randn(problem.num_locations + 1, 2) * 0.1 + [16.5, 80.5]
    
    distance_matrix = np.zeros((problem.num_locations + 1, problem.num_locations + 1))
    for i in range(problem.num_locations + 1):
        for j in range(i + 1, problem.num_locations + 1):
            dist = np.linalg.norm(coords[i] - coords[j])
            distance_matrix[i, j] = distance_matrix[j, i] = dist
            
    routes, distances = quantum_solver.solve_quantum_vrp(
        distance_matrix,
        problem.num_vehicles,
        depot_node,
        reps=problem.reps
    )
    
    return {
        "routes": routes,
        "distances": distances,
        "coordinates": coords.tolist()
    }