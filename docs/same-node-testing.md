# Same-Node Client/Server Testing (Kubernetes)

This runbook pins the client pod to the same node as the `object-detection` server
to minimize network variance and stabilize latency measurements.

## 1 Confirm the server pod is running

```bash
kubectl get pods -n ml-pipelines -o wide
```

Note the server pod name and the `NODE` it is running on.

## 2 Update the test script ConfigMap

```bash
kubectl create configmap -n ml-pipelines od-test-image \
  --from-file=test_image.py=tests/test_image.py \
  --dry-run=client -o yaml | kubectl apply -f -
```

## 3 Run the client pod on the same node

Replace `NODE_NAME_HERE` with the server pod node (from step 1).

```bash
kubectl delete pod -n ml-pipelines od-client
```

```bash
kubectl run -n ml-pipelines od-client --image=python:3.11-slim \
  --overrides='{
    "spec": {
      "nodeSelector": { "kubernetes.io/hostname": "rci-tide-gpu-01.sdsu.edu" },
      "containers": [{
        "name": "od-client",
        "image": "python:3.11-slim",
        "command": ["bash","-lc"],
        "args": ["pip -q install requests && pip install aiohttp && OD_BASE_URL=http://object-detection.ml-pipelines.svc.cluster.local:8000 python /scripts/test_image.py; sleep 3600"],
        "volumeMounts": [{"name":"scripts","mountPath":"/scripts"}]
      }],
      "volumes": [{"name":"scripts","configMap":{"name":"od-test-image"}}]
    }
  }'
```

## 4 Run the test inside the client pod

```bash
kubectl exec -n ml-pipelines -it od-client -- bash
```

Inside the pod:

```bash
OD_BASE_URL=http://object-detection.ml-pipelines.svc.cluster.local:8000 python /scripts/test_image.py
```

Or one-shot without an interactive shell:

```bash
kubectl exec -n ml-pipelines od-client -- bash -lc \
  "OD_BASE_URL=http://object-detection.ml-pipelines.svc.cluster.local:8000 python /scripts/test_image.py"
```

## 5 Check placement (optional)

```bash
kubectl get pods -n ml-pipelines -o wide
```

## Cleanup

```bash
kubectl delete pod -n ml-pipelines od-client
kubectl delete configmap -n ml-pipelines od-test-image
```
