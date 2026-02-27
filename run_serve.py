"""Startup script to run Ray Serve with correct host binding for Docker."""
import subprocess
import time
import ray
from ray import serve

# Start Ray head node with a fixed metrics export port for Prometheus scraping
subprocess.run(
    ["ray", "start", "--head", "--dashboard-host=0.0.0.0", "--metrics-export-port=8080"],
    check=True,
)

# Wait for Ray head node to be ready
time.sleep(5)

# Connect to the running Ray instance
ray.init(address="auto", ignore_reinit_error=True)

# Start Serve with HTTP binding to all interfaces
serve.start(http_options={"host": "0.0.0.0", "port": 8000})

# Import and run the model
from src.object_detection.object_detection import entrypoint
serve.run(entrypoint, name="default", route_prefix="/")

# Block forever
while True:
    time.sleep(3600)
