Alexander Lio & Ashwin Senthilvasan

Deployment on nautilus kubernetes:

```bash
brew install kubectl

#OIDC Authentication
brew install int128/kubelogin/kubelogin
```
Download config file from nrp.ai documentation

```bash
#to force authentication
kubectl auth whoami 

mkdir -p ~/.kube

mv ~/Downloads/config ~/.kube/config

kubectl config set-context --current --namespace=ml-pipelines
```

this sets up your namespace 

```bash
# Deploy 
kubectl apply -f k8s/deployment.yaml

# Check pod status while waiting for the container to start
kubectl get pods -n ml-pipelines -w

# View logs on what is happening on deploymeny (verify also that you are using cluster resources)
kubectl logs -n ml-pipelines deployment/object-detection

# Port forward to actually run tests on client end
kubectl port-forward -n ml-pipelines svc/object-detection 8000:8000

# Delete deployment (after testing or want to build new code)
kubectl delete -f k8s/deployment.yaml

# Check all resources
kubectl get all -n ml-pipelines
```
## Structure

- `k8s/deployment.yaml` → Kubernetes deployment and service configuration
- `Dockerfile` → Builds the Docker image with CUDA/GPU support for x86_64 only
- `.github/workflows/docker-build.yml` → GitHub Actions to build and push image to Docker Hub
- `run_serve.py` → Custom startup script for Docker containers
- `src/object_detection/object_detection.py` → Ray Serve application with YOLOv5

## Code Changes from Original Branch

### Why `run_serve.py` exists

The original code used `serve run src.object_detection.object_detection:entrypoint` to start the server. This works on local but fails in Docker containers because Ray Serve binds to `127.0.0.1` (localhost) by default.

`run_serve.py` fixes this with:
```python
ray.init(dashboard_host="0.0.0.0", ignore_reinit_error=True)
serve.start(http_options={"host": "0.0.0.0", "port": 8000})
```

This connects the HTTP server to (`0.0.0.0`), allowing access through port-forwarding.

### Changes to `object_detection.py`


- Removed `ray.init()` (now handled by `run_serve.py` in Docker, or automatically by `serve run` on local dev)
- Added `/detect-upload` POST endpoint for direct image upload which performed better
- Added `health_check_period_s=60` to reduce periodic latency spikes

