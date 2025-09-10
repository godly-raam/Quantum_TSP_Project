Q-Fleet: A Quantum-Hybrid VRP Optimization Engine
Q-Fleet is a full-stack application developed for the Amaravati Quantum Valley Hackathon 2025. It provides a "quantum-ready" framework for solving the complex Vehicle Routing Problem (VRP) using a hybrid quantum-classical approach. The project features a Python backend powered by Qiskit's QAOA algorithm and a modern, interactive frontend built with a JavaScript framework and deployed on Vercel.

📋 Features
Interactive UI: A clean, web-based dashboard to define VRP instances (number of vehicles, delivery locations).

Quantum Backend: Leverages the Quantum Approximate Optimization Algorithm (QAOA) to find optimal routes.

Problem Mapping: Automatically converts the real-world VRP into a Quadratic Unconstrained Binary Optimization (QUBO) problem suitable for quantum solvers.

Tunable Algorithm: Allows users to adjust the QAOA reps (quality) parameter to see the trade-off between runtime and solution accuracy.

Interactive Visualization: Displays the final, optimized routes for each vehicle on an interactive Folium map.

API-based Architecture: Decoupled frontend and backend for a professional, scalable solution.

🏛️ Project Architecture
This project uses a modern client-server architecture. The frontend is a static web app hosted on Vercel, and the backend is a Python API hosted as a web service on Render.

Licensed by Google

Frontend (Vercel): The user interacts with the web app, defines a VRP, and clicks "Optimize".

API Request: The frontend sends a POST request with the problem data to the backend API.

Backend (Render):

The FastAPI server receives the request.

It generates a distance matrix and formulates a QUBO.

The Qiskit Sampler and QAOA algorithm find the optimal solution.

API Response: The backend returns the optimized routes and coordinates to the frontend as a JSON object.

Visualization: The frontend uses the received data to draw the routes on the interactive map.

🚀 Getting Started
Prerequisites
A free account on Render for the backend.

A free account on Vercel for the frontend.

A GitHub account.

Python 3.10+ installed locally.

Backend Setup (Render)
Fork this Repository: Create a copy of this project in your own GitHub account.

Create a New Web Service on Render:

Connect your forked GitHub repository.

Runtime: Python 3

Build Command: pip install -r requirements.txt

Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT

Deploy: Click "Create Web Service". After deployment, Render will provide you with a public URL (Soon).

Frontend Setup (Vercel)
Clone the frontend repository (provided by your frontend developer).

In the frontend's code, locate the API call and replace the placeholder URL with your live Render backend URL (Soon).

Push the changes to the frontend's GitHub repository.

Connect this repository to Vercel to deploy the website.

🛠️ Technologies Used
Quantum Backend: Qiskit, Qiskit Aer, Qiskit Algorithms, FastAPI, Python

Frontend: JavaScript/TypeScript, React/Next.js/Vue, Vercel

Deployment: Render, GitHub

Visualization: Folium
