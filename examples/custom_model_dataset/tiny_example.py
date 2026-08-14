"""Example custom --model-id/--dataset-id target (docs/coding_guidelines.md 5.C "Model
Agnostic"): proves a user's own architecture/dataset plugs into
AutoCIM-Agent with zero changes to this repo, via tools/qat.py's
"module.path:function_name" spec.

Run against these two (no download, CPU-fast, real gradient descent on
tiny synthetic data -- expect very low/meaningless accuracy, this is a
wiring smoke test, not a real research candidate):

    python main.py --model-id examples.custom_model_dataset.tiny_example:build_model \\
        --dataset-id examples.custom_model_dataset.tiny_example:load_dataset \\
        --hw-config examples/hw_configs/default_cim_v1_128x128.json

`build_model`/`load_dataset` are the two contracts tools/qat.py's
`_resolve_model_builder`/`_resolve_dataset_loader` require of any custom
spec -- copy this file's shape for a real architecture/dataset.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def build_model(num_classes: int) -> nn.Module:
    """Callable[[int], nn.Module] -- called once with the loaded dataset's
    real num_classes (from load_dataset() below), never a hardcoded value."""
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, num_classes),
    )


def load_dataset() -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Callable[[], (train_loader, val_loader, test_loader, num_classes)]
    -- three disjoint splits, exactly like tools/qat.py's built-in CIFAR10
    loader. Replace this synthetic random data with your own real dataset;
    everything downstream (fake-quant, pruning, training loop) is already
    dataset-agnostic."""
    num_classes = 3

    def _split(size: int) -> DataLoader:
        images = torch.randn(size, 3, 16, 16)
        labels = torch.randint(0, num_classes, (size,))
        return DataLoader(TensorDataset(images, labels), batch_size=8)

    return _split(64), _split(16), _split(16), num_classes
