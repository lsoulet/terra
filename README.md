# terra

Satellite land-use classification with distributed processing: split large Sentinel-2 scenes into tiles, classify tiles in parallel across Ray workers, reassemble into a colored land-use map.

## Prerequisites

- Docker, with your user in the `docker` group (`sudo usermod -aG docker $USER`, then reconnect your session for it to take effect)
- An NVIDIA GPU with the driver installed on the host (`nvidia-smi` should work)
- Docker's CDI device support enabled — GPU access uses `--device=nvidia.com/gpu=all` rather than the older `--gpus all` flag. If `--gpus`-based commands fail with `AMD CDI spec not found` (a red herring — no AMD hardware involved), run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` first (this restarts the Docker daemon, stopping any running containers).

## Two environments

This repo runs on two separate, deliberately distinct environments:

| Environment | Purpose | Filesystem | Jupyter |
|---|---|---|---|
| **Docker** (`terra-ray`) | Day-to-day dev, fast iteration | Live bind mount of the repo | `http://127.0.0.1:8888/lab` |
| **KubeRay** (minikube) | Production-like testing (e.g. `RayJob` runs) | Baked into the image at build time — rebuild + reload to update | `http://127.0.0.1:8890/lab` (via `kubectl port-forward`) |

The intended flow: prototype interactively in the Docker Jupyter, then once code is ready, submit it as a script against the KubeRay cluster (e.g. via `RayJob`) to validate it in a more production-shaped setting.

In application code, always connect with `ray.init(address="auto")` rather than a bare `ray.init()` — this attaches to whatever cluster is already running (Docker's single-node cluster or the KubeRay-managed one) without any code changes.

## Tests and CI

`tests/` covers the pure, CPU-only helper functions in each script (`calibrate()`, `majority_filter()`, `s3_to_https()`) — the parts that are both deterministic and don't need a GPU, a Ray cluster, or real network access, which rules out testing the scripts end-to-end here.

```bash
pip install -r requirements-dev.txt
pytest -v
ruff check .
```

`requirements-dev.txt` is deliberately separate from `requirements.txt` — it skips torch and its bundled CUDA runtime (most of the production image's ~10GB), since nothing the tests import needs them.

`.github/workflows/ci.yml` runs lint (`ruff`), the unit tests (`pytest`), and YAML validation (`yamllint infrastructure/`) on every push/PR; the Docker image itself is only rebuilt when `Dockerfile`/`requirements.txt`/`entrypoint.sh` actually change.

---

## Experiment tracking (MLflow)

`distributed_tiling.py`, `distributed_inference.py`, `calibrate_reflectance.py`, and `train_hparam_search.py` all log params/metrics/artifacts to a local MLflow tracking store at `data/outputs/mlruns/mlflow.db` (SQLite backend — MLflow 3.x's plain filesystem store is deprecated). Each script owns its own experiment:

| Experiment | Script | What it tracks |
|---|---|---|
| `terra-distributed-tiling` | `distributed_tiling.py` | tile counts, throughput |
| `terra-land-use-inference` | `distributed_inference.py` | class distribution, smoothing effect |
| `terra-reflectance-calibration` | `calibrate_reflectance.py` | `REFLECTANCE_MAX` search vs. EuroSAT's own pixel distribution |
| `terra-hparam-search` | `train_hparam_search.py` | per-epoch train/val curves for each hyperparameter config |

### UI

```bash
docker run -d --rm \
  --name terra-mlflow \
  -p 5000:5000 \
  -v $(pwd):/app \
  terra:latest \
  sh -c "mlflow ui --backend-store-uri sqlite:///data/outputs/mlruns/mlflow.db --host 0.0.0.0 --port 5000"
```

Open `http://localhost:5000`. Select runs in an experiment and click **Compare** to chart a metric against a param across all of them — e.g. `distance_to_eurosat` vs. `reflectance_max` in `terra-reflectance-calibration`, or the per-epoch `val_acc` curves in `terra-hparam-search`.

**Gotcha — stale file handle**: this container holds `mlflow.db` open for as long as it runs. If that file ever gets deleted and recreated (e.g. wiping `data/outputs/mlruns/` to reset tracking), the container keeps serving the old, now-unlinked file instead of the new one — new runs silently stop appearing in the UI. Fix: `docker restart terra-mlflow`.

**Gotcha — Docker and KubeRay don't share this store**: the UI above reads the host bind-mount (`data/outputs/` in the repo). KubeRay's `terra-tiles-pvc` is a *separate* volume backed by a directory inside the minikube VM, not the host repo — so a `RayJob` run on KubeRay writes to a different `mlflow.db` entirely, invisible to the UI above. `infrastructure/mlflow.yaml` deploys the same UI as its own pod on the KubeRay side, with the PVC mounted, so it can read that store instead:

```bash
kubectl apply -f infrastructure/mlflow.yaml
kubectl port-forward svc/terra-mlflow-svc 5001:5000
```

**Gotcha — default memory limit OOM-kills this pod**: `mlflow ui` spawns 4 uvicorn worker processes by default, using more memory than the Docker container (which has no memory limit) ever revealed — `2Gi` gets the pod `OOMKilled` within a minute of startup, with no error beyond a silently-restarting pod and connection-refused errors on port-forward. `mlflow.yaml` sets `4Gi`, which is stable.

---

## Docker environment

### Build

```bash
docker build -t terra:latest .
```

The image installs `requirements.txt` (including `torch` with bundled CUDA runtime and `ray[default]`) on top of `python:3.12-slim`, and includes the trained checkpoint (`models/resnet18_eurosat_best.pt`, ~43MB — small enough to bake in directly). No CUDA toolkit is baked into the image — only the host's NVIDIA driver and CDI passthrough are required. Larger, regenerable data (`data/eurosat/`, `data/sentinel2_scene/`, `data/outputs/`) stays excluded via `.dockerignore`.

### Run

```bash
docker run -it --rm \
  --name terra-ray \
  --device=nvidia.com/gpu=all \
  -p 8888:8888 \
  -p 8265:8265 \
  -v $(pwd):/app \
  terra:latest
```

The container's entrypoint (`entrypoint.sh`) starts a single-node Ray cluster (`ray start --head`) and then launches Jupyter Lab. The whole repo is bind-mounted at `/app`, so notebooks, `data/`, and `models/` are all live without rebuilding the image.

| Container port | Purpose |
|---|---|
| 8888 | Jupyter Lab |
| 8265 | Ray dashboard |
| 6379 | Ray GCS (internal only — not published; only needed if an external worker joins the cluster) |

If you're working over VSCode Remote-SSH, forward ports 8888 and 8265 from the "PORTS" panel (`Ctrl+Shift+P` → "Forward a Port") to reach them from your local browser.

### Test the cluster

1. **Jupyter** — open the `http://127.0.0.1:8888/lab?token=...` URL printed in the container logs (`docker logs terra-ray`).
2. **Ray dashboard** — open `http://localhost:8265`.
3. **Cluster smoke test**:
   ```bash
   docker exec terra-ray python scripts/test_ray_cluster.py
   ```
   (or from a Jupyter cell: `%run scripts/test_ray_cluster.py`). It checks that `ray.init(address="auto")` connects, that tasks actually run in parallel, and that a GPU-declared Ray actor can see the GPU via `torch.cuda.is_available()`.

### Distributed tiling

`distributed_tiling.py` splits a Sentinel-2 L1C scene into 64x64 tiles in parallel with Ray, saved as `.npy` files under `data/outputs/sentinel2_tiles/` (gitignored/dockerignored — generated data, not committed). It's self-contained: it looks up a low-cloud scene itself via the public Earth Search STAC API and reads each band directly over HTTPS (no local download needed — see `data/pull_sentinel_scene.py` instead if you just want a local copy of the scene to explore in a notebook).

```bash
docker exec terra-ray python scripts/distributed_tiling.py                # full scene, ~29,000 tiles, ~8 min
docker exec terra-ray python scripts/distributed_tiling.py --max-rows 30  # demo-sized, ~5,000 tiles, ~1 min
```

### Distributed inference

`distributed_inference.py` classifies those tiles with the Phase 1 EuroSAT ResNet18, using a Ray actor that loads the model once and streams batches through it rather than reloading it per tile. Raw reflectance is calibrated with one fixed scale (not a per-tile stretch — see the script's docstring for why that matters) before normalization, then a 3x3 majority filter smooths the result into coherent regions.

```bash
docker exec terra-ray python scripts/distributed_inference.py
```

Outputs (`land_use_grid.npy`, `land_use_grid_smoothed.npy`) are saved under `data/outputs/land_use_maps/` — see `notebooks/05_land_use_map.ipynb` for visualizing the result.

### Reflectance calibration search

`calibrate_reflectance.py` picks `REFLECTANCE_MAX` (the fixed constant `distributed_inference.py` divides raw tiles by before feeding them to the model) by matching a sample of real tiles' mean/std to EuroSAT's own — a label-free proxy for calibration quality, since there's no ground truth for the real scene. Not Ray-parallelized: it's a small local computation on already-downloaded tiles, no meaningful distributed-systems problem to justify it.

```bash
docker exec terra-ray python scripts/calibrate_reflectance.py
```

Prints and logs (to `terra-reflectance-calibration`) the distance-to-EuroSAT for each candidate value; the best one gets copied into `REFLECTANCE_MAX` in `distributed_inference.py` by hand.

### Hyperparameter search

`train_hparam_search.py` reruns notebook 2's fine-tuning loop with several hyperparameter configs at once, comparing them under identical data/seed conditions. There's only one physical GPU on this cluster, so "at once" means several Ray tasks sharing it through a fractional allocation (`num_gpus=0.25` each) rather than true multi-GPU parallelism — CUDA doesn't isolate that budget, so they genuinely run concurrently on the device.

```bash
docker exec terra-ray python scripts/train_hparam_search.py
```

Prints a ranked summary (best `val_acc` first) and logs every config, with its full per-epoch curve, to `terra-hparam-search`.

### Operations

```bash
# Status
docker ps -a --filter name=terra-ray

# Start
docker start terra-ray

# Stop
docker stop terra-ray

# Restart
docker restart terra-ray
```

---

## Kubernetes / KubeRay environment

A local Kubernetes cluster (minikube) running a `RayCluster` via the KubeRay operator, with its own Jupyter pod — separate from the Docker environment above.

### Setup

```bash
minikube start --driver=docker --gpus=all --cpus=8 --memory=16g
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator --version 1.6.2 -f infrastructure/kuberay-operator-values.yaml
minikube image load terra:latest
kubectl apply -f infrastructure/tiles-pvc.yaml
kubectl apply -f infrastructure/ray-cluster.yaml
kubectl apply -f infrastructure/jupyter.yaml
kubectl apply -f infrastructure/mlflow.yaml
kubectl port-forward svc/terra-jupyter-svc 8890:8888
kubectl port-forward svc/terra-mlflow-svc 5001:5000
```

`tiles-pvc.yaml` is a shared volume (mounted at the same path on the head, worker, and Jupyter pods) for `distributed_tiling.py`'s output — without it, each pod writes tiles to its own local disk, and a `RayJob` can report `SUCCEEDED` while the output ends up scattered across pods with no single place containing all of it.

### Updating the image

After code changes, rebuild and point the cluster at the fresh image:

```bash
docker build -t terra:latest .
minikube image load terra:latest
kubectl delete pods -l ray.io/cluster=terra-ray-cluster -l app=terra-jupyter
```

Two gotchas, both requiring the pod-deletion step above (and sometimes more) to actually take effect:
- `minikube image load` silently keeps the old image if a container is still using that tag — pods must be deleted so they get recreated on the new one.
- Even after that, the kubelet can still report a stale `imageID` for a freshly-created pod (a caching layer independent of both Docker and the CRI tool). If pods keep running old code despite the steps above, restart the kubelet directly: `minikube ssh -- "sudo systemctl restart kubelet"`, then delete the pods again.

### Test the cluster

```bash
kubectl delete rayjob terra-test-job --ignore-not-found
kubectl apply -f infrastructure/ray-job-test.yaml
kubectl get rayjob terra-test-job -w
kubectl logs -l job-name=terra-test-job
```

Checks the same things as the Docker smoke test (`test_ray_cluster.py`), but submitted as a `RayJob` against the persistent `terra-ray-cluster`.

### Distributed tiling

Same script as the Docker environment, submitted as a `RayJob` (runs with `--max-rows 30` by default — see `infrastructure/ray-job-tiling.yaml`):

```bash
kubectl delete rayjob terra-tiling-job --ignore-not-found
kubectl apply -f infrastructure/ray-job-tiling.yaml
kubectl get rayjob terra-tiling-job -w
kubectl logs -l job-name=terra-tiling-job
```

A `SUCCEEDED` status only means no task raised an exception — it doesn't guarantee the output is complete or in one place. Verify the tile count directly:

```bash
kubectl exec deploy/terra-jupyter -- sh -c "ls data/outputs/sentinel2_tiles/*.npy | wc -l"
```

### Distributed inference

Same script as the Docker environment, submitted as a `RayJob` (`infrastructure/ray-job-inference.yaml`):

```bash
kubectl delete rayjob terra-inference-job --ignore-not-found
kubectl apply -f infrastructure/ray-job-inference.yaml
kubectl get rayjob terra-inference-job -w
kubectl logs -l job-name=terra-inference-job
```

Outputs land in `data/outputs/land_use_maps/` — already covered by the same PVC as the tiles (`tiles-pvc.yaml`), so no extra volume needed. Verify from any pod:

```bash
kubectl exec deploy/terra-jupyter -- sh -c "ls data/outputs/land_use_maps/"
```

### Operations

```bash
# Status
minikube status
kubectl get raycluster
kubectl get pods
kubectl get rayjob

# Start
minikube start
kubectl port-forward svc/terra-jupyter-svc 8890:8888 &

# Stop
minikube stop

# Restart
minikube stop && minikube start
```
