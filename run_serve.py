"""Startup script to run Ray Serve with correct host binding for Docker."""
import ray
from ray import serve

# Initialize Ray
ray.init(dashboard_host="0.0.0.0", ignore_reinit_error=True)

# Start Serve with HTTP binding to all interfaces
serve.start(http_options={"host": "0.0.0.0", "port": 8000})

# Import and run the model
from src.object_detection.object_detection import entrypoint
serve.run(entrypoint, name="default", route_prefix="/")

# Block forever
import time
while True:
    time.sleep(3600)
