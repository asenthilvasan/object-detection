#!/bin/bash

# Cleanup function to stop all services
cleanup() {
    echo ""
    echo "Shutting down services..."
    pkill -9 -f "serve run" 2>/dev/null
    pkill -9 -f "ray::" 2>/dev/null
    pkill -9 -f "gcs_server" 2>/dev/null
    pkill -9 -f "raylet" 2>/dev/null
    poetry run ray stop --force 2>/dev/null
    brew services stop grafana 2>/dev/null
    echo "Done."
    exit 0
}

# Trap Ctrl+C (SIGINT) and call cleanup
trap cleanup SIGINT SIGTERM

# Start Grafana
brew services start grafana

# Start Prometheus
poetry run ray metrics launch-prometheus

# Start Ray Serve in background
poetry run serve run src.object_detection.object_detection:entrypoint &
SERVE_PID=$!

# Wait for serve to exit (this allows trap to work)
wait $SERVE_PID

# When Ray Serve exits normally, also cleanup
cleanup

# Summary of whats running: 
# Grafana: http://localhost:3000 (your imported dashboards live here)
# Prometheus: http://localhost:9090
# Ray Serve: http://localhost:8000
# Ray Dashboard: http://localhost:8265