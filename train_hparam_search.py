"""Compare hyperparameter configurations for the Phase 1 EuroSAT ResNet18,
in parallel, on a single physical GPU.

There's only one GPU on this cluster (see ray-cluster.yaml), so "parallel"
here doesn't mean multiple physical devices -- it means multiple Ray tasks
sharing that one GPU through a fractional allocation (num_gpus=0.25 instead
of 1), the same mechanism as distributed_inference.py's TileClassifier just
split four ways. CUDA doesn't isolate that budget, it's a Ray scheduling
hint -- the tasks genuinely run concurrently on the device, which is fine
here: ResNet18 at 64x64 is tiny next to this GPU's 24GB.

Each config trains for SEARCH_EPOCHS (fewer than the notebook's full run --
this is a ranking pass, not a final model) and returns its history to the
driver, which does all MLflow logging itself. Logging from inside the tasks
would mean multiple processes writing to the same SQLite file concurrently;
routing results back through Ray's RPC (like distributed_tiling.py's
per-row byte counts) avoids that entirely.
"""

import time
from pathlib import Path

import mlflow
import numpy as np
import ray
from sklearn.model_selection import train_test_split

# Same OUTPUTS_DIR convention as the other scripts -- data/outputs is the PVC
# mount point on KubeRay. EuroSAT (224MB) lives here too, not just under
# data/eurosat/, so it survives GPU pod restarts instead of being
# re-downloaded from scratch each time (data/eurosat/ was the notebook's
# original location, kept as-is; this is a copy, not a move).
#
# EUROSAT_ROOT is the *parent* passed as `root=` to torchvision's EuroSAT --
# it expects (and creates, if download=True) an `eurosat/2750/<class>/*.jpg`
# structure underneath whatever root it's given, so root itself is
# data/outputs, not data/outputs/eurosat.
OUTPUTS_DIR = Path("data/outputs")
EUROSAT_ROOT = OUTPUTS_DIR
STATS_PATH = Path("data/eurosat_stats.json")
MLFLOW_TRACKING_DIR = OUTPUTS_DIR / "mlruns"
MLFLOW_EXPERIMENT = "terra-hparam-search"
CHECKPOINT_DIR = Path("models")

SEARCH_EPOCHS = 10
BATCH_SIZE = 64
SEED = 42

# Same 3 axes discussed: learning rate, optimizer, and how much of the
# pretrained backbone stays trainable. "baseline" reproduces notebook 2's
# config exactly, so its result here is a sanity check against the known
# 97.43% -- not directly comparable (5 epochs here vs. 10 there), but should
# track the same shape of learning curve.
CONFIGS = [
    {"name": "baseline", "optimizer": "adam", "lr": 1e-4, "freeze_backbone": False, "weight_decay": 0.0},
    {"name": "higher_lr", "optimizer": "adam", "lr": 3e-4, "freeze_backbone": False, "weight_decay": 0.0},
    {"name": "frozen_backbone", "optimizer": "adam", "lr": 1e-4, "freeze_backbone": True, "weight_decay": 0.0},
    {"name": "sgd_momentum", "optimizer": "sgd", "lr": 1e-2, "freeze_backbone": False, "weight_decay": 1e-4, "momentum": 0.9},
]


@ray.remote(num_gpus=0.25, num_cpus=1)
def train_config(config):
    # Deferred imports: same reason as TileClassifier in distributed_inference.py
    # -- importing torch at module level breaks Ray's serialization of this
    # function's closure.
    import json
    import random

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms
    from torchvision.datasets import EuroSAT
    from torchvision.models import ResNet18_Weights, resnet18

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(STATS_PATH) as f:
        stats = json.load(f)
    mean, std = stats["mean"], stats["std"]

    # download=True only re-downloads if EUROSAT_ROOT is missing/incomplete --
    # here it's pre-populated, so this just verifies and moves on.
    base = EuroSAT(root=EUROSAT_ROOT, download=True)
    labels = [label for _, label in base.samples]
    indices = list(range(len(base)))

    # Same stratified 70/15/15 split as notebook 2, same seed -- every config
    # trains and validates on the identical split, so differences in the
    # results are attributable to the hyperparameters, not to a lucky split.
    train_idx, val_idx = train_test_split(
        indices, test_size=0.3, stratify=labels, random_state=SEED,
    )

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = Subset(EuroSAT(root=EUROSAT_ROOT, transform=train_transform), train_idx)
    val_set = Subset(EuroSAT(root=EUROSAT_ROOT, transform=eval_transform), val_idx)

    # num_workers kept low: up to 4 of these tasks run at once on a 4-CPU
    # node, so each only gets a thin slice regardless.
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(base.classes))
    if config["freeze_backbone"]:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")
    model = model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(trainable, lr=config["lr"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            trainable, lr=config["lr"], momentum=config.get("momentum", 0.0),
            weight_decay=config["weight_decay"],
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    criterion = nn.CrossEntropyLoss()

    def run_epoch(loader, train):
        model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for images, targets in loader:
                images, targets = images.to(device), targets.to(device)
                if train:
                    optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                if train:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(dim=1) == targets).sum().item()
                total += images.size(0)
        return total_loss / total, correct / total

    history = []
    best_val_acc = 0.0
    start = time.time()
    for epoch in range(1, SEARCH_EPOCHS + 1):
        train_loss, train_acc = run_epoch(train_loader, train=True)
        val_loss, val_acc = run_epoch(val_loader, train=False)
        best_val_acc = max(best_val_acc, val_acc)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })
    elapsed = time.time() - start

    return config, best_val_acc, history, elapsed


if __name__ == "__main__":
    ray.init(address="auto")

    MLFLOW_TRACKING_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_TRACKING_DIR.resolve()}/mlflow.db")

    client = mlflow.MlflowClient()
    if client.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        client.create_experiment(
            MLFLOW_EXPERIMENT,
            artifact_location=str(MLFLOW_TRACKING_DIR.resolve() / "artifacts"),
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print(f"Launching {len(CONFIGS)} configs concurrently (0.25 GPU each)...")
    futures = [train_config.remote(cfg) for cfg in CONFIGS]
    results = ray.get(futures)

    print(f"\n{'config':20s} {'best_val_acc':>12s} {'elapsed':>10s}")
    for config, best_val_acc, history, elapsed in sorted(results, key=lambda r: -r[1]):
        print(f"{config['name']:20s} {best_val_acc:12.4f} {elapsed:9.1f}s")

        with mlflow.start_run(run_name=config["name"]):
            mlflow.log_params(config)
            mlflow.log_param("search_epochs", SEARCH_EPOCHS)
            for h in history:
                mlflow.log_metrics(
                    {"train_loss": h["train_loss"], "train_acc": h["train_acc"],
                     "val_loss": h["val_loss"], "val_acc": h["val_acc"]},
                    step=h["epoch"],
                )
            mlflow.log_metric("best_val_acc", best_val_acc)
            mlflow.log_metric("elapsed_seconds", elapsed)

    winner = max(results, key=lambda r: r[1])
    print(f"\nBest config: {winner[0]['name']} (val_acc={winner[1]:.4f})")
    print("Re-run the winner for the full NUM_EPOCHS in notebook 2 to produce the final checkpoint.")
