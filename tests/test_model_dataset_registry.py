"""tools/qat.py: runtime model/dataset registration + "module:function"
custom specs (docs/coding_guidelines.md C "Model Agnostic" -- @tuner must handle
arbitrary nn.Module/datasets, not just the three built-in model_ids).

No network/download involved: every custom builder/loader here is a tiny
synthetic model + synthetic random data, matching this suite's existing
`_TinyConvNet`/`make_fake_qat_backend` style (tests/conftest.py).
"""

from typing import Tuple

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tools.qat import _DATASET_REGISTRY, _MODEL_REGISTRY, get_layer_groups, get_qat_backend, register_dataset, register_model


class _CustomTinyModel(nn.Module):
    """Two real top-level stages ("stem"/"head") so group_quantizable_layers
    has something non-trivial to group, distinct from conftest.py's
    _TinyConvNet so the two test suites' registries never collide."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.stem(x))
        x = self.pool(x).flatten(1)
        return self.head(x)


def _custom_model_builder(num_classes: int) -> nn.Module:
    """Module-level (importable) builder -- the target of this test file's
    own "module:function" spec, exercising the exact dynamic-import path a
    real user's own module would go through (tools.qat._import_callable)."""
    return _CustomTinyModel(num_classes)


def _custom_dataset_loader() -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Module-level (importable) zero-argument dataset loader -- tiny
    synthetic data, 3 disjoint splits, num_classes=5 (deliberately not 10,
    to prove num_classes really flows from the dataset, not a hardcoded
    default)."""

    def _split(size: int) -> DataLoader:
        images = torch.randn(size, 3, 8, 8)
        labels = torch.randint(0, 5, (size,))
        return DataLoader(TensorDataset(images, labels), batch_size=4)

    return _split(12), _split(4), _split(4), 5


_CUSTOM_MODEL_SPEC = f"{__name__}:_custom_model_builder"
_CUSTOM_DATASET_SPEC = f"{__name__}:_custom_dataset_loader"


def test_module_function_spec_resolves_custom_model():
    groups = get_layer_groups(_CUSTOM_MODEL_SPEC, num_classes=5)
    assert set(groups.keys()) == {"stem", "head"}


def test_register_model_then_lookup_by_plain_id():
    # register_model mutates the module-level _MODEL_REGISTRY -- other
    # tests (tests/test_qat.py's exact-equality check against the 3
    # built-ins) share that same global, so this must clean up after
    # itself regardless of outcome, not just assert and return.
    try:
        register_model("test_registered_model_xyz", _custom_model_builder)
        groups = get_layer_groups("test_registered_model_xyz", num_classes=5)
        assert set(groups.keys()) == {"stem", "head"}
    finally:
        _MODEL_REGISTRY.pop("test_registered_model_xyz", None)


def test_unknown_model_id_raises_with_supported_list():
    with pytest.raises(ValueError, match="resnet18"):
        get_layer_groups("no_such_model_id_and_no_colon")


def test_malformed_module_function_spec_raises():
    with pytest.raises(ValueError, match="module.path:function_name"):
        get_layer_groups("this_looks_like_a_spec:")


def test_module_function_spec_resolves_custom_dataset_and_num_classes_flows_through():
    backend = get_qat_backend(_CUSTOM_MODEL_SPEC, dataset_id=_CUSTOM_DATASET_SPEC)
    assert backend.num_classes == 5
    images, _ = next(iter(backend.train_loader))
    logits = backend.base_model(images)
    assert logits.shape[-1] == 5


def test_register_dataset_then_lookup_by_plain_id():
    # Same global-mutation cleanup concern as test_register_model_then_
    # lookup_by_plain_id above, for _DATASET_REGISTRY.
    try:
        register_dataset("test_registered_dataset_xyz", _custom_dataset_loader)
        backend = get_qat_backend(_CUSTOM_MODEL_SPEC, dataset_id="test_registered_dataset_xyz")
        assert backend.num_classes == 5
    finally:
        _DATASET_REGISTRY.pop("test_registered_dataset_xyz", None)


def test_unknown_dataset_id_raises_with_supported_list():
    with pytest.raises(ValueError, match="cifar10"):
        get_qat_backend(_CUSTOM_MODEL_SPEC, dataset_id="no_such_dataset_id_and_no_colon")
