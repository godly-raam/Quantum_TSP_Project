# main.py - Complete CORS Fix for OPTIONS requests

import os
import sys
import logging
import time
from typing import Optional, Dict, Any
sys.path.append('./modules')

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import numpy as np
from modules import quantum_solver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Q-Fleet API",
    description="A production-grade, quantum-hybrid Vehicle Routing Problem solver.",
    version="1.0.0"
)

# --- COMPREHENSIVE CORS FIX ---
# This configuration handles all CORS issues including preflight OPTIONS requests

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=False,  # Must be False when using "*"
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # Cache preflight for 10 minutes
)

# Additional manual CORS handling for stubborn preflight requests
@app.middleware("http")
async def cors_handler(request: Request, call_next):
    # Handle preflight OPTIONS requests manually
    if request.method == "OPTIONS":
        response = JSONResponse(content={})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Max-Age"] = "600"
        return response
    
    # Process normal requests
    response = await call_next(request)
    
    # Add CORS headers to all responses
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    
    return response

# Pydantic models for robust API contracts
class VrpProblem(BaseModel):
    num_locations: int = Field(..., ge=2, le=6, description="Number of locations (2-6)")
    num_vehicles: int = Field(..., ge=1, le=3, description="Number of vehicles (1-3)")
    reps: int = Field(4, ge=1, le=6, description="QAOA depth (1-6)")

class VrpResponse(BaseModel):
    routes: list
    distances: list
    coordinates: list
    total_distance: float
    solution_method: str
    execution_time: float
    is_quantum_solution: bool
    notes: str

# Explicit OPTIONS handler for all routes (fallback)
@app.options("/{path:path}")
async def options_handler(path: str):
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "600",
        }
    )

@app.get("/api/health")
def health_check():
    logger.info("Health check endpoint was called.")
    return {"status": "healthy", "message": "Q-Fleet API is running!"}

@app.post("/api/optimize", response_model=VrpResponse)
def optimize_routes(problem: VrpProblem):
    logger.info(f"Received VRP request: {problem.dict()}")
    try:
        np.random.seed(123)
        depot_node = 0
        coords = np.random.randn(problem.num_locations + 1, 2) * 0.1 + [16.5, 80.5]
        
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
        logger.error(f"Error in optimize_routes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")