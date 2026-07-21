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