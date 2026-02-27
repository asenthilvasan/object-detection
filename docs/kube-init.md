

# Initial Setup for Nautilus and Kubernetes

## Authors: Alexander Lio & Ashwin Senthilvasan

We will show the step-by-step but official instructions for cluster access can be found at: https://nrp.ai/documentation/userdocs/start/getting-started/

first install kubectl and kubelogin

```bash
brew install kubectl # one time command

#OIDC Authentication
brew install int128/kubelogin/kubelogin # one time command
```
then download the config file at this link:
- https://nrp.ai/config
- IMPORTANT: save it to `$HOME/.kube` 
    - if this doesnt exist, create it using `mkdir ~/.kube`

finally, the file should be saved at `$HOME/.kube/config`

next, do authentication and set namespace
```bash
#to force authentication will bring up browser and you login 
kubectl auth whoami 

kubectl config set-context --current --namespace=ml-pipelines # or whatever your namespace may be called
```
At this point, you can now run the deployment.

### Helpful Commands

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

