# terra

Satellite land-use classification with distributed processing: split large Sentinel-2 scenes into tiles, classify tiles in parallel across Ray workers, reassemble into a colored land-use map.

## Prerequisites

- Docker, with your user in the `docker` group (`sudo usermod -aG docker $USER`, then reconnect your session for it to take effect)
- An NVIDIA GPU with the driver installed on the host (`nvidia-smi` should work)
- Docker's CDI device support enabled — GPU access uses `--device=nvidia.com/gpu=all` rather than the older `--gpus all` flag

## Building the image

```bash
docker build -t terra:latest .
```

The image installs `requirements.txt` (including `torch` with bundled CUDA runtime and `ray[default]`) on top of `python:3.12-slim`. No CUDA toolkit is baked into the image — only the host's NVIDIA driver and CDI passthrough are required.

## Running the stack

```bash
docker run -it --rm \
  --device=nvidia.com/gpu=all \
  -p 8888:8888 \
  -p 8265:8265 \
  -v $(pwd):/app \
  terra:latest
```

The container's entrypoint (`entrypoint.sh`) starts a single-node Ray cluster (`ray start --head`) and then launches Jupyter Lab. The whole repo is bind-mounted at `/app`, so notebooks, `data/`, and `models/` are all live without rebuilding the image.

### Ports

| Container port | Purpose |
|---|---|
| 8888 | Jupyter Lab |
| 8265 | Ray dashboard |
| 6379 | Ray GCS (internal only — not published; only needed if an external worker joins the cluster) |

If you're working over VSCode Remote-SSH, forward ports 8888 and 8265 from the "PORTS" panel (`Ctrl+Shift+P` → "Forward a Port") to reach them from your local browser.

## Testing the cluster

Once the container is running, check:

1. **Jupyter** — open the `http://127.0.0.1:8888/lab?token=...` URL printed in the container logs (`docker logs <container>`).
2. **Ray dashboard** — open `http://localhost:8265`.
3. **Cluster smoke test** — run `test_ray_cluster.py`, either from a terminal:
   ```bash
   docker exec <container> python test_ray_cluster.py
   ```
   or from a Jupyter cell (`%run test_ray_cluster.py`). It checks that `ray.init(address="auto")` connects, that tasks actually run in parallel, and that a GPU-declared Ray actor can see the GPU via `torch.cuda.is_available()`.

In application code, always connect with `ray.init(address="auto")` rather than a bare `ray.init()` — this attaches to whatever cluster is already running (local single-node today, a future KubeRay-managed cluster on Kubernetes) without any code changes.

## Kubernetes / KubeRay (production-like testing)

`infrastructure/` holds a second, separate environment: a local Kubernetes cluster (minikube) running a `RayCluster` via the KubeRay operator, with its own Jupyter pod. It exists alongside the Docker environment above, deliberately, with a distinct purpose:

| Environment | Purpose | Filesystem | Jupyter |
|---|---|---|---|
| Docker (`terra-ray`) | Day-to-day dev, fast iteration | Live bind mount of the repo | `http://127.0.0.1:8888/lab` |
| KubeRay (minikube) | Production-like testing (e.g. `RayJob` runs) | Baked into the image at build time — rebuild + `minikube image load` to update | `http://127.0.0.1:8890/lab` (via `kubectl port-forward`) |

The intended flow: prototype interactively in the Docker Jupyter, then once code is ready, submit it as a script against the KubeRay cluster (e.g. via `RayJob`) to validate it in a more production-shaped setting — not the other way around, and not by running two copies of the same notebook workflow.

### Setup

```bash
minikube start --driver=docker --gpus=all --cpus=8 --memory=16g
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator --version 1.6.2 -f infrastructure/kuberay-operator-values.yaml
minikube image load terra:latest
kubectl apply -f infrastructure/ray-cluster.yaml
kubectl apply -f infrastructure/jupyter.yaml
kubectl port-forward svc/terra-jupyter-svc 8890:8888
```

Note: `docker run --gpus all` (and therefore `minikube start --gpus=all`) requires a `nvidia` runtime registered with Docker — if it fails with `AMD CDI spec not found`, run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` first (this restarts the Docker daemon, stopping any running containers, including `terra-ray`).

### Updating the image

After code changes, the KubeRay pods need to be pointed at a fresh image — reloading alone is not enough if pods are already running on the old one:

```bash
docker build -t terra:latest .
minikube image load terra:latest
kubectl delete pods -l ray.io/cluster=terra-ray-cluster -l app=terra-jupyter
```

`minikube image load` silently keeps the old image if a container is still using that tag, so pods must be deleted (they get recreated automatically) for the new image to actually take effect.

### Operations cheat sheet

```bash
# Status
docker ps -a --filter name=terra-ray
minikube status
kubectl get raycluster
kubectl get pods
kubectl get rayjob

# Start
docker start terra-ray
minikube start
kubectl port-forward svc/terra-jupyter-svc 8890:8888 &

# Stop
docker stop terra-ray
minikube stop

# Restart
docker restart terra-ray
minikube stop && minikube start

# Run a test job against the "prod" (KubeRay) cluster
kubectl delete rayjob terra-test-job --ignore-not-found
kubectl apply -f infrastructure/ray-job-test.yaml
kubectl get rayjob terra-test-job -w
kubectl logs -l job-name=terra-test-job
```