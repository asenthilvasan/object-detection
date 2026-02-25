# Prometheus + Grafana + k6 Load Testing (Kubernetes)

This runbook sets up:
1. **Prometheus** – scrapes Ray metrics from the object-detection pod
2. **Grafana** – auto-provisions the official Ray-provided dashboards
3. **k6** – generates artificial load against `/detect-upload`

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  ml-pipelines namespace          │
│                                                  │
│  ┌──────────────────┐   :8080/metrics            │
│  │  object-detection│◄──────────────────────┐   │
│  │  (Ray Serve)     │                        │   │
│  └──────────────────┘                        │   │
│                                              │   │
│  ┌──────────────┐   scrape :9090   ┌─────────┴─┐ │
│  │   Grafana    │◄────────────────►│ Prometheus │ │
│  │  :3000       │                  │  :9090     │ │
│  └──────────────┘                  └───────────┘ │
│                                                  │
│  ┌──────────────┐                                │
│  │  k6 Job      │──► POST /detect-upload :8000   │
│  └──────────────┘                                │
└─────────────────────────────────────────────────┘
```

Ray exposes metrics on **port 8080** (`/metrics`) automatically when `ray[serve]` is running.
The `deployment.yaml` pod template already has the `prometheus.io/scrape` annotations set.

---

## 1. Deploy Prometheus

```bash
kubectl apply -f k8s/prometheus.yaml
```

Verify it is scraping Ray:

```bash
# Port-forward Prometheus UI
kubectl port-forward -n ml-pipelines svc/prometheus 9090:9090

# Open http://localhost:9090
# Go to Status → Targets – you should see the "ray" job as UP
# Try querying: ray_serve_num_ongoing_requests_total
```

---

## 2. Deploy Grafana

```bash
kubectl apply -f k8s/grafana.yaml
```

Grafana uses an **init container** to download the four official Ray dashboard JSON files
from the Ray GitHub repo before the main container starts:

| Dashboard | What it shows |
|-----------|---------------|
| `default_grafana_dashboard.json` | Ray Core – node/actor/task metrics |
| `serve_grafana_dashboard.json` | Ray Serve – replica count, queue depth |
| `serve_deployment_grafana_dashboard.json` | Per-deployment latency & throughput |
| `data_grafana_dashboard.json` | Ray Data pipeline metrics |

Wait for the init container to finish -> must wait till the init container says "Running", then port-forward:

```bash
kubectl port-forward -n ml-pipelines svc/grafana 3000:3000
```
Wait a couple of minutes for Grafana to fully initialize, then open **http://localhost:3000** (login: `admin` / `admin`).
Navigate to **Dashboards → Ray** to see all four dashboards pre-loaded.

> **Note:** The Prometheus datasource is pre-configured automatically via provisioning.
> No manual setup is needed inside Grafana.

---

## 3. Run k6 Load Tests

### One-shot Job (recommended)

```bash
# Apply the ConfigMap + Job
kubectl apply -f k8s/k6-load-test.yaml

# Watch the job run
kubectl get job -n ml-pipelines k6-load-test -w -> kind of whatever

# Stream k6 output
kubectl logs -n ml-pipelines -l app=k6-load-test -f
```

The default load profile uses the `ramping-arrival-rate` executor, which controls
**requests per second** directly rather than concurrent VUs:

| Stage | Duration | Target RPS |
|-------|----------|------------|
| Ramp-up | 30 s | 0 → 10 RPS |
| Sustained | 2 min | 10 RPS |
| Ramp-up | 30 s | 10 → 40 RPS |
| Sustained | 2 min | 40 RPS |
| Ramp-up | 30 s | 40 → 80 RPS |
| Sustained | 2 min | 80 RPS |
| Ramp-down | 30 s | 80 → 0 RPS |

**Thresholds** (job fails if breached):
- `p(95)` inference latency < 5 s
- Error rate < 5 %

> **VU sizing note:** `preAllocatedVUs` (default: 50) should be ≥ `target_rps × p99_latency_s`.
> For example, at 80 RPS with ~2 s p99 latency you need ~160 VUs minimum. Increase
> `preAllocatedVUs` / `maxVUs` in the ConfigMap if k6 logs a "insufficient VUs" warning.

### Re-running the Job

Kubernetes Jobs are immutable once created. To re-run:

### Deleting the job is important !!

```bash
kubectl delete job -n ml-pipelines k6-load-test 
kubectl apply -f k8s/k6-load-test.yaml
```

### Tuning load

Edit the `stages` block inside the `rps_ramp` scenario in `k8s/k6-load-test.yaml`.
`target` is **requests per second**:

```js
stages: [
  { duration: "30s", target: 10 },  // ramp 0 → 10 RPS
  { duration: "2m",  target: 10 },  // hold  10 RPS
  { duration: "30s", target: 40 },  // ramp 10 → 40 RPS
  { duration: "2m",  target: 40 },  // hold  40 RPS
  { duration: "30s", target: 80 },  // ramp 40 → 80 RPS
  { duration: "2m",  target: 80 },  // hold  80 RPS
  { duration: "30s", target: 0  },  // ramp down
],
```

Increase `target` values or add more stages to stress the system harder.

### Same-node placement

To pin k6 to the same node as the object-detection pod (eliminates cross-node network
variance), uncomment the `nodeSelector` block in `k8s/k6-load-test.yaml` and set the
correct hostname (see `docs/same-node-testing.md` for how to find the node name).

---

## 4. Watching metrics during load

With both port-forwards active:

- **Grafana → Ray → Serve Deployment Dashboard**: watch `num_ongoing_requests`,
  `request_latency_ms`, `replica_starts_total` update in real time as k6 ramps up.
- **Grafana → Ray → Default Dashboard**: watch CPU/memory/actor counts.
- **Prometheus → Graph**: ad-hoc PromQL queries like:
  ```promql
  histogram_quantile(0.95, rate(ray_serve_deployment_request_latency_ms_bucket[1m]))
  ```

---

## Cleanup

```bash
kubectl delete -f k8s/k6-load-test.yaml
kubectl delete -f k8s/grafana.yaml
kubectl delete -f k8s/prometheus.yaml
```

### The above commands do technically delete the PVC but not sure if it is consistent
- in your terminal output from the commands above, look for ```persistentvolumeclaim "prometheus-data" deleted from ml-pipelines namespace```
- if you see this, then the PVC is deleted
- if you don't see this, then the PVC is not deleted and you need to delete it manually

> PVCs (`prometheus-data`, `grafana-data`) are **not** deleted by the above commands.
> Delete them explicitly if you want to reclaim storage:
> ```bash
> kubectl delete pvc -n ml-pipelines prometheus-data grafana-data
> ```
