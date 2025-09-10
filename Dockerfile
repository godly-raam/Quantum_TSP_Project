# Dockerfile

# Use a standard, lightweight Python image
FROM python:3.11-slim

# Install the necessary system-level compilers (this is the fix)
RUN apt-get update && apt-get install -y build-essential gfortran

# Set the working directory inside the container
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port the app will run on
EXPOSE 10000

# The command to run your app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]