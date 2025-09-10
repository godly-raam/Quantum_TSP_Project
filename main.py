# main.py

import os
import sys
import logging
import time
from typing import Optional, Dict, Any
sys.path.append('./modules')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import numpy as np
from modules import quantum_solver

# Configure professional logging for Render
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Q-Fleet API",
    description="A production-grade, quantum-hybrid Vehicle Routing Problem solver.",
    version="1.0.0"
)

# CORS Middleware allows your Vercel frontend to connect
origins = [
    "https://entangled-minds-qc.vercel.app",
    "https://*.vercel.app", # Allows Vercel preview deployments
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for robust API contracts
class VrpProblem(BaseModel):
    num_locations: int = Field(..., ge=2, le=8, description="Number of locations (2-8)")
    num_vehicles: int = Field(..., ge=1, le=4, description="Number of vehicles (1-4)")
    reps: int = Field(5, ge=1, le=8, description="QAOA Quality (repetitions)")

class VrpResponse(BaseModel):
    routes: list
    distances: list
    coordinates: list
    total_distance: float
    solution_method: str
    execution_time: float
    is_quantum_solution: bool
    notes: str

@app.middleware("http")
async def add_process_time_header(request, call_next):
    """This middleware adds a custom header to the response to measure total processing time."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 3))
    return response

# The health check endpoint is used by Render to ensure your service is running.
@app.get("/api/health")
def health_check():
    """A health check endpoint for Render to monitor the service."""
    logger.info("Health check endpoint was called.")
    return {"status": "healthy", "message": "Q-Fleet API is running!"}

@app.post("/api/optimize", response_model=VrpResponse)
def optimize_routes(problem: VrpProblem):
    """This is the main endpoint that solves the VRP."""
    logger.info(f"Received VRP request: {problem.dict()}")
    try:
        np.random.seed(123)
        depot_node = 0
        coords = np.random.randn(problem.num_locations + 1, 2) * 0.1 + [16.5, 80.5]
        
        # *** THIS IS THE CORRECTED LINE ***
        distance_matrix = np.zeros((problem.num_locations + 1, problem.num_locations + 1))
        for i in range(problem.num_locations + 1):
            for j in range(i + 1, problem.num_locations + 1):
                dist = np.linalg.norm(coords[i] - coords[j])
                distance_matrix[i, j] = distance_matrix[j, i] = dist

        routes, distances, metrics = quantum_solver.solve_quantum_vrp(
            distance_matrix,
            problem.num_vehicles,
            depot_node,
            reps=problem.reps
        )
        
        solution_method = "Quantum QAOA" if metrics.is_valid_quantum_solution else "Classical Fallback"

        return {
            "routes": routes,
            "distances": distances,
            "coordinates": coords.tolist(),
            "total_distance": metrics.total_distance,
            "solution_method": solution_method,
            "execution_time": metrics.execution_time,
            "is_quantum_solution": metrics.is_valid_quantum_solution,
            "notes": metrics.notes
        }
    except Exception as e:
        # This will catch any unexpected errors and return a clean error to the frontend
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
