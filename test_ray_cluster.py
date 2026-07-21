import time

import ray
import torch

ray.init(address="auto")

print("Cluster resources:", ray.cluster_resources())


@ray.remote
def square(x):
    time.sleep(1)
    return x * x


start = time.time()
results = ray.get([square.remote(i) for i in range(8)])
elapsed = time.time() - start
print(f"Parallel task results: {results} (took {elapsed:.1f}s, would take 8s sequentially)")


@ray.remote(num_gpus=1)
class GPUWorker:
    def check_gpu(self):
        return torch.cuda.is_available(), torch.cuda.get_device_name(0)


worker = GPUWorker.remote()
available, name = ray.get(worker.check_gpu.remote())
print(f"GPU actor sees CUDA: {available} ({name})")
