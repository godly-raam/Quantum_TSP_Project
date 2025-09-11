# main.py - Final Fix for CORS
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Q-Fleet API",
    description="A production-grade, quantum-hybrid Vehicle Routing Problem solver.",
    version="1.0.0"
)

# --- CORRECTED CORS CONFIGURATION ---
# Get environment variables for better security
environment = os.getenv("ENVIRONMENT", "development")
frontend_url = os.getenv("FRONTEND_URL", "https://entangled-minds-qc.vercel.app")

# Base origins
origins = [
    frontend_url,  # Your main Vercel app
    "https://entangled-minds-qc.vercel.app",  # Explicit main URL
]

# Add development origins if not in production
if environment != "production":
    origins.extend([
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ])

# Add common Vercel preview deployment patterns
# Note: Wildcard doesn't work, so we need to handle this differently
vercel_patterns = [
    "https://entangled-minds-qc-git-main-your-username.vercel.app",  # Replace with your actual git pattern
    "https://entangled-minds-qc-git-develop-your-username.vercel.app",  # Add other branches as needed
]
origins.extend(vercel_patterns)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
    ],
)

# Alternative: More permissive CORS for development (USE CAREFULLY)
# Uncomment this block and comment the above if you need temporary broader access
"""
if environment == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Only for development!
        allow_credentials=False,  # Must be False when using allow_origins=["*"]
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
"""

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
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 3))
    
    # Add CORS headers manually for additional coverage
    response.headers["Access-Control-Allow-Origin"] = request.headers.get("origin", "*")
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Accept, Accept-Language, Content-Language, Content-Type, Authorization, X-Requested-With"
    
    return response



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
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")