Alexander Lio & Ashwin Senthilvasan

Deployment on nautilus kubernetes:

```bash
brew install kubectl

#OIDC Authentication
brew install int128/kubelogin/kubelogin
```
Download config file from nrp.ai documentation

```bash
#to force authentication will bring up browser and you login 
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
kubectl get pods -n ml-pipelines
```
## Structure

- `k8s/deployment.yaml` → Kubernetes deployment and service configuration
- `Dockerfile` → Builds the Docker image with CUDA/GPU support for x86_64 only
- `.github/workflows/docker-build.yml` → GitHub Actions to build and push image to Docker Hub
- `run_serve.py` → Custom startup script for Docker containers
- `src/object_detection/object_detection.py` → Ray Serve application with YOLOv5

