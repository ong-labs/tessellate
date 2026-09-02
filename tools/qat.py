"""Real PyTorch QAT engine for @tuner.

The column-wise weight + partial-sum quantization arrangement below is not
this project's idea: it follows Kim et al., "Column-wise Quantization of
Weights and Partial Sums for Accurate and Efficient Compute-In-Memory
Accelerators" (DATE 2025, arXiv:2502.07842). What this file implements is a
simplification of it -- notably WITHOUT that paper's learnable (LSQ) scale
factors and without its array-level tiling of partial sums. Do not describe
this file as an implementation of that paper.

Replaces the earlier accuracy-by-formula mock with genuine PyTorch
mechanics: real crossbar-aligned fake-quantization (straight-through
estimator) and real magnitude-ranked structured column pruning, applied to
a real pretrained model, fine-tuned with real gradient descent, and
evaluated for real measured accuracy.

Per-layer granularity: `get_layer_groups` exposes the target model's real
top-level module names (e.g. resnet18's `conv1`/`layer1`/`layer2`/`layer3`/
`layer4`/`fc`), grouping every Conv2d/Linear submodule under whichever real
stage it belongs to. @planner (nodes/planner.py) reads this to propose a
`LayerBitConfig` per real stage instead of one flat entry for the whole
model; `apply_crossbar_quant_prune` then applies each stage's own
(weight_bits, column_pruning_ratio) to only that stage's layers, falling
back to `default_config` (the request's averaged config) for any stage
@planner didn't cover -- so an LLM response that only names a subset of
stages, or names none at all (e.g. `@tuner`'s own single-entry fallback,
nodes/common.py's `default_layer_configs`), still degrades gracefully to
uniform quantization rather than erroring.

Scope, stated plainly:
- Device-agnostic: `run_qat_tuning` moves its (deep-copied, per-trial)
  model and each batch to `_resolve_device`'s result -- an explicit
  `TESSELLATE_QAT_DEVICE` override, else auto-detected (CUDA, then Apple
  MPS, then CPU) -- never hardcoded to assume a GPU exists, since most
  CI/dev machines don't (docs/coding_guidelines.md 5.C). `get_qat_backend`'s cached
  `base_model`/loaders stay on CPU regardless -- only the per-trial deep
  copy moves, so the shared cache is never pinned to one device.
- `_MODEL_REGISTRY` ships three built-in entries: "resnet18" (torchvision,
  CNN), "mobilenet_v2" (torchvision, depthwise-separable CNN), and
  "vit_tiny" (timm's `vit_tiny_patch16_224`, transformer) -- all
  ImageNet-pretrained, final classifier head replaced for the loaded
  dataset's real class count (see `_DATASET_REGISTRY` below, not a fixed
  10 anymore). Any other `model_id` is resolved two ways, in order:
  `register_model(model_id, builder)` at runtime, or -- with zero changes
  to this file -- a `"module.path:function_name"` import spec (a
  caller's own architecture, defined in their own module) matching
  `Callable[[int], nn.Module]`, same contract as the built-ins
  (docs/coding_guidelines.md 5.C "Model Agnostic": arbitrary `nn.Module` support, not
  just three hardcoded names).
- Datasets are the same two-tier story via `_DATASET_REGISTRY`: the
  built-in `"cifar10"` (default when `dataset_id` is omitted), a
  `register_dataset(dataset_id, loader)` runtime hook, or a
  `"module.path:function_name"` spec resolving to a zero-argument
  `Callable[[], (train_loader, val_loader, test_loader, num_classes)]`
  -- a deployment's real target dataset (ImageNet, a proprietary
  dataset, ...) plugs in the same way, entirely outside this repo.
- Hardware-mapping honesty: a CIM crossbar is fundamentally a dense
  matrix-vector multiply engine. ViT/DeiT's attention (qkv/proj) and MLP
  (fc1/fc2) layers are *all* dense `Linear` ops -- a natural fit. MobileNetV2's
  depthwise-separable convolutions include per-channel depthwise convs where
  each output channel only touches one input channel, so they don't exercise
  a crossbar's parallel-MAC advantage the way a dense/pointwise layer does.
  `apply_crossbar_quant_prune` still quantizes/prunes every Conv2d/Linear
  uniformly regardless of this distinction (a reasonable simplification for
  first-cut multi-architecture coverage) -- `tools/cim_physics.py`'s
  mapper/profiler modeling depthwise vs. dense MACs differently would be a
  real follow-up, not attempted here.
- Fine-tuning/eval default (`dataset_id` omitted) to a fixed CIFAR10 subset
  (not the full 50k/10k images), so a QAT trial completes in CPU time by
  default -- an explicit,
  approved scope decision, not a hidden shortcut, and one a deployment can
  widen (up to the full split) via `TESSELLATE_QAT_TRAIN_SIZE`/`_VAL_SIZE`/
  `_TEST_SIZE`/`_BATCH_SIZE`, or train for longer per candidate via
  `TESSELLATE_QAT_MAX_EPOCHS` (`resolve_max_epochs`, the single source of truth
  `nodes/tuner.py`/`tools/batch_warmup.py` both call), without a code
  change (docs/coding_guidelines.md 5.C). Whatever
  the configured sizes, `_DEFAULT_TRAIN_SIZE`/`_VAL_SIZE`/`_TEST_SIZE` are a
  genuine three-way split (see `QATBackend`'s docstring): training and
  early-stopping never touch the test set that `run_qat_tuning`
  reports accuracy on, so that number isn't leaked into the decisions that
  produced it. Still a subset, not the full benchmark -- comparing candidate
  configs against each other is sound; treating the absolute number as a
  publishable benchmark result is not.
- Grouping (`_stage_of`) is by top-level attribute name, extended through
  the first purely-numeric path segment if there is one -- e.g. resnet18's
  "layer2.0.conv1" groups as "layer2.0" (one real stage per residual block),
  and vit_tiny's "blocks.7.attn.qkv" groups as "blocks.7" (one real stage
  per transformer block), not everything under "blocks" collapsing into a
  single stage. Real architectural granularity, tractable for both the LLM
  prompt and tools/search.py's search space, rather than either a single
  flat model-wide entry or a much higher-dimensional per-individual-layer
  search.

`get_qat_backend`/`get_layer_groups` are the swappable seams tests replace
(tests/conftest.py) with a tiny synthetic model + synthetic data, exactly
like `llm.py`'s `get_planner_chat_model` -- no test in this suite downloads
a dataset/checkpoint or trains on real data.
"""

from __future__ import annotations

import copy
import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.utils.data import DataLoader

_CACHE_ROOT = Path(__file__).resolve().parent.parent / ".cache"

StageConfig = Tuple[int, float]  # (weight_bits, column_pruning_ratio)


@dataclass
class QATBackend:
    """A model + its fine-tuning/eval data, bundled so `run_qat_tuning` is
    agnostic to whether it's driving the real resnet18/CIFAR10 backend or a
    test double.

    Three disjoint loaders, not two: `val_loader` is what `run_qat_tuning`
    monitors for early stopping, and `test_loader` is only ever touched once,
    after training/early-stopping decisions are final, for the accuracy
    actually reported. Using the same set for both (an earlier version of
    this module did) leaks the "test" set into the training-decision loop --
    the reported accuracy would be mildly optimistic, biased toward whatever
    that set happens to look like, rather than a genuinely held-out number.
    """

    base_model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    layer_groups: Dict[str, List[str]]


# ---------------------------------------------------------------------------
# Real per-stage layer grouping
# ---------------------------------------------------------------------------


def _stage_of(module_name: str) -> str:
    """Groups by the dotted name up through the first purely-numeric path
    segment, if any -- that segment is always a `ModuleList`/`Sequential`
    child index, i.e. one repeating architectural block. Architectures that
    give each stage its own top-level attribute (resnet18's `layer1`..`layer4`)
    still resolve to per-residual-block groups ("layer2.0", "layer2.1", ...),
    and architectures that nest *all* their blocks under one container
    attribute (mobilenet_v2's `features`, vit_tiny's `blocks`) resolve to one
    group per real block ("features.3", "blocks.7", ...) instead of every
    block collapsing into a single "features"/"blocks" group -- which a plain
    "just take the first segment" rule would do."""
    parts = module_name.split(".")
    for index, part in enumerate(parts):
        if part.isdigit():
            return ".".join(parts[: index + 1])
    return parts[0]


def group_quantizable_layers(model: nn.Module) -> Dict[str, List[str]]:
    """Every Conv2d/Linear submodule's dotted name, grouped by real stage.
    Pure function of `model` (no model_id/registry lookup) so it's directly
    unit-testable against an arbitrary small module -- no network/dataset
    involved."""
    groups: Dict[str, List[str]] = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            groups.setdefault(_stage_of(name), []).append(name)
    return groups


# ---------------------------------------------------------------------------
# Real crossbar-aligned fake-quantization + structured column pruning
# ---------------------------------------------------------------------------


class _FakeQuantSTE(torch.autograd.Function):
    """Symmetric, per-output-channel ("per-crossbar-column") fake
    quantization with a straight-through gradient estimator -- each output
    channel (dim 0, same "crossbar column" convention `_CrossbarQuantPrune`
    below already prunes at) gets its own scale, not one scale shared
    across the whole layer.

    A single per-tensor scale sizes the entire quantization grid off
    whichever column happens to have the largest weight magnitude,
    crushing every other column's own dynamic range down to a handful of
    quantization steps near zero -- real crossbar columns routinely differ
    by an order of magnitude or more in weight scale. Per-column scaling
    is standard practice for accurate CIM quantization at a given
    bit-width -- it's also the more physically accurate model here, not
    just an accuracy trick: each crossbar column has its own physical
    read-out path, so an independent scale per column matches the
    hardware being simulated more closely than one shared scale would.

    Still one bit-width per stage (`LayerBitConfig.weight_bits`, unchanged)
    -- only the quantization grid's *scale* is now per-column; forward
    simulates the resulting grid, backward passes the gradient through
    unchanged so the optimizer can still update the underlying
    full-precision weight.

    `channel_dim` generalizes which axis is "the column": weight tensors
    are `[out_channels, in_channels, ...]` (dim 0), but a layer's *output*
    tensor -- used by `_PartialSumQuant` below to quantize the analog
    partial sum a real ADC would read out, before it leaves the array --
    is `[batch, out_channels, ...]` (dim 1). Same math either way, just a
    different axis moved to the front before the per-channel reduction."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, bits: int, channel_dim: int = 0) -> torch.Tensor:
        qmax = max(2 ** (bits - 1) - 1, 0)
        qmin = -(2 ** (bits - 1))
        moved = x.detach().movedim(channel_dim, 0)
        num_channels = moved.shape[0]
        per_channel_max = moved.reshape(num_channels, -1).abs().max(dim=1).values
        scale_shape = [1] * x.dim()
        scale_shape[channel_dim] = num_channels
        scale = per_channel_max.clamp(min=1e-8).view(scale_shape) / max(qmax, 1)
        return torch.clamp(torch.round(x / scale), qmin, qmax) * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None


class _CrossbarQuantPrune(nn.Module):
    """`torch.nn.utils.parametrize` transform applied to a layer's `weight`:
    fake-quantize to `bits` (`_FakeQuantSTE`, per-column scale), then zero
    the `pruning_ratio` fraction of output channels ("crossbar columns",
    dim 0 in PyTorch's [out_channels, in_channels, ...] weight layout) with
    the smallest L1 magnitude -- a standard structured-pruning ranking
    criterion. Quantization happens first, so pruning ranks (and later
    zeroes) already-quantized columns, matching the order a real crossbar
    would apply them in.
    """

    def __init__(self, bits: int, pruning_ratio: float):
        super().__init__()
        self.bits = bits
        self.pruning_ratio = pruning_ratio

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        w = _FakeQuantSTE.apply(weight, self.bits, 0)
        out_channels = w.shape[0]
        k = int(round(out_channels * self.pruning_ratio))
        if k > 0:
            l1_per_channel = w.detach().reshape(out_channels, -1).abs().sum(dim=1)
            prune_idx = torch.argsort(l1_per_channel)[:k]
            mask = torch.ones(out_channels, dtype=w.dtype, device=w.device)
            mask[prune_idx] = 0.0
            w = w * mask.view(-1, *([1] * (w.dim() - 1)))
        return w


class _PartialSumQuant:
    """Forward hook (not a parametrization -- there's no `weight` to
    attach to; this quantizes the module's *output*) applied to a
    Conv2d/Linear's output: fake-quantizes it to `adc_bits`, one scale per
    output channel. A layer's raw output here stands in for the analog
    partial sum a real crossbar accumulates per column before its ADC
    reads it out -- the ADC's own limited resolution is what actually
    bounds how precisely that accumulation is captured, independent of
    however many bits the weights/inputs were quantized to. Per-column
    scale for the same reason `_CrossbarQuantPrune` uses one for weights:
    different output channels can have very different accumulated
    magnitudes, and one shared scale would waste ADC codes on whichever
    channel happens to run hottest.

    `adc_bits` is `HWConfig.adc_bits` -- a fixed physical property of the
    chip (docs/coding_guidelines.md 5.C: never hardcoded, always sourced from HWConfig),
    not a per-stage value `@planner`'s search tunes the way weight_bits is.
    A single crossbar's ADC serves every column of that array the same
    way; there's no per-stage knob to turn here in real hardware, so
    `apply_crossbar_quant_prune` takes it once for the whole model rather
    than per stage_configs entry.

    dim 1 (not dim 0) is the channel axis here: a module's output is
    `[batch, out_channels, ...]`, batch first -- `_FakeQuantSTE`'s
    `channel_dim` parameter exists specifically so this and
    `_CrossbarQuantPrune` (dim 0, no batch axis on a weight tensor) can
    share the same underlying math."""

    def __init__(self, adc_bits: int):
        self.adc_bits = adc_bits

    def __call__(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        return _FakeQuantSTE.apply(output, self.adc_bits, 1)


def apply_crossbar_quant_prune(
    model: nn.Module,
    stage_configs: Dict[str, StageConfig],
    default_config: StageConfig,
    adc_bits: Optional[int] = None,
) -> None:
    """Mutates `model` in place: registers `_CrossbarQuantPrune` on every
    `Conv2d`/`Linear` submodule's `weight`, using that module's real stage's
    entry in `stage_configs` (keyed by `LayerBitConfig.layer_name` ==
    `group_quantizable_layers`'s stage names) if present, else
    `default_config`. Callers must pass a model they own (e.g. a fresh
    `copy.deepcopy` of a cached base model), never the shared cached
    instance -- see `run_qat_tuning`.

    `adc_bits`, if given, also registers `_PartialSumQuant` as a forward
    hook on every one of those same submodules -- column-wise partial-sum
    quantization (see its docstring), applied identically to every stage
    since ADC precision is a hardware-wide constant, not something
    stage_configs varies. `None` (the default) skips it entirely, matching
    every caller's behavior before this parameter existed."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            bits, pruning_ratio = stage_configs.get(_stage_of(name), default_config)
            parametrize.register_parametrization(module, "weight", _CrossbarQuantPrune(bits, pruning_ratio))
            if adc_bits is not None:
                module.register_forward_hook(_PartialSumQuant(adc_bits))


# ---------------------------------------------------------------------------
# Device selection (docs/coding_guidelines.md 5.C: config-driven, never hardcoded)
# ---------------------------------------------------------------------------

_DEVICE_ENV = "TESSELLATE_QAT_DEVICE"


def _resolve_device(requested: Optional[str] = None) -> torch.device:
    """`requested` (or `TESSELLATE_QAT_DEVICE` if `requested` is None) wins
    when set; otherwise auto-detects the best available accelerator --
    CUDA, then Apple's MPS backend, then CPU. Never raises on a machine
    with no GPU (that's the common case: CI, most dev/deployment
    machines) -- silently resolves to CPU rather than requiring every
    caller to know whether a GPU exists."""
    candidate = requested if requested is not None else os.environ.get(_DEVICE_ENV)
    if candidate:
        return torch.device(candidate)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Fine-tuning + evaluation loop
# ---------------------------------------------------------------------------


def _train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion, device: torch.device
) -> None:
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss_sum += criterion(outputs, labels).item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return {"accuracy": correct / total if total else 0.0, "loss": loss_sum / total if total else 0.0}


def run_qat_tuning(
    backend: QATBackend,
    stage_configs: Dict[str, StageConfig],
    default_config: StageConfig,
    max_epochs: int,
    early_stopping_patience: Optional[int] = None,
    device: Optional[str] = None,
    adc_bits: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Real QAT trial for one candidate config: deep-copies `backend`'s base
    model (so each candidate starts from the same pretrained weights, never
    a previous candidate's fine-tuned weights), moves that copy to
    `_resolve_device(device)` (`backend.base_model`/loaders themselves stay
    on CPU -- only this per-trial copy moves), applies each real stage's
    own crossbar-aligned fake-quant + structured pruning (falling back to
    `default_config` for any stage `stage_configs` doesn't cover) plus
    `adc_bits`-precision column-wise partial-sum quantization on every
    stage alike if `adc_bits` is given (`apply_crossbar_quant_prune`),
    fine-tunes for up to `max_epochs` real gradient-descent epochs against
    `backend.train_loader` (monitoring `backend.val_loader` for early
    stopping), and returns real measured accuracy on `backend.test_loader`
    -- touched exactly once, after every training/early-stopping decision is
    already final, so the reported number isn't leaked into those decisions.

    Two things `early_stopping_patience` deliberately does *not* do. It
    never restores the best-validation-loss weights: the returned accuracy
    is always the last epoch trained, whether the loop ran out of epochs or
    stopped early, since no checkpoint is kept. And it is off in every
    production path -- `tools/simulators.py`'s `tuner_tool` leaves
    `TunerToolInput.early_stopping_patience` at its `None` default, so a
    real session always trains the full `max_epochs`
    (`resolve_max_epochs()`); the parameter exists for callers that pass it
    explicitly (tests). Adding best-weight restore would change every
    accuracy this project has already measured, so it is a deliberate
    deferral rather than an oversight.

    Thread-safety caveat, deliberately not papered over: `seed` below seeds
    torch's **global** RNG, and `backend.train_loader` is a `shuffle=True`
    DataLoader with no explicit generator, so its per-epoch permutation is
    drawn from that same global RNG. Concurrent trials therefore do not get
    independent, reproducible streams -- `tools/batch_warmup.py`'s thread
    pool has each worker reseed the global RNG mid-flight of the others, so
    with more than one worker a candidate's accuracy is not reproducible
    from `(stage_configs, seed)` alone. One worker (the default on a single
    CUDA device, for the separate reason batch_warmup documents) has no such
    race. Making this genuinely thread-safe means giving each trial its own
    generator-backed train loader, which changes the permutation every
    existing result was measured under -- a deliberate deferral, not an
    oversight.

    `seed`, if given, seeds torch's global RNG right before training starts
    -- `backend.base_model`'s own weights are deterministic (a cached,
    already-initialized instance, deep-copied unchanged every call), so the
    only real per-trial stochasticity this can reach is `train_loader`'s
    batch shuffling and any dropout in the architecture. Without a seed
    (`None`, the default -- unchanged behavior for any existing caller),
    that stochasticity is genuinely random every call, so even the exact
    same `stage_configs` on the exact same candidate can't be reproduced
    -- `ExecutionContext.run_seed` (schemas/config.py) is the intended
    source, threaded in by `tools/simulators.py`'s `tuner_tool`.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    resolved_device = _resolve_device(device)
    model = copy.deepcopy(backend.base_model).to(resolved_device)
    apply_crossbar_quant_prune(model, stage_configs, default_config, adc_bits)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    epochs_run = 0
    for epoch in range(max_epochs):
        _train_one_epoch(model, backend.train_loader, optimizer, criterion, resolved_device)
        epochs_run = epoch + 1
        val_metrics = _evaluate(model, backend.val_loader, criterion, resolved_device)
        if val_metrics["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            break

    final_metrics = _evaluate(model, backend.test_loader, criterion, resolved_device)
    return {"accuracy": final_metrics["accuracy"], "epochs_run": epochs_run, "device": str(resolved_device)}


# ---------------------------------------------------------------------------
# Default backend: real resnet18 (ImageNet-pretrained) + real CIFAR10 subset
# ---------------------------------------------------------------------------


def _build_resnet18(num_classes: int) -> nn.Module:
    import torchvision.models as tv_models

    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_mobilenet_v2(num_classes: int) -> nn.Module:
    import torchvision.models as tv_models

    model = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def _build_vit_tiny(num_classes: int) -> nn.Module:
    import timm

    # timm's `num_classes=` kwarg replaces the pretrained head itself
    # (unlike torchvision's models, there's no separate manual-surgery step).
    return timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=num_classes)


_MODEL_REGISTRY: Dict[str, Callable[[int], nn.Module]] = {
    "resnet18": _build_resnet18,
    "mobilenet_v2": _build_mobilenet_v2,
    "vit_tiny": _build_vit_tiny,
}

# Structural-only default: only used to build a throwaway instance for
# get_layer_groups() when no real dataset (and thus no real num_classes)
# is in play yet -- a classifier head's output width never changes a
# module's *name*, which is all group_quantizable_layers reads. A real
# get_qat_backend() call always overrides this with the loaded dataset's
# actual num_classes (_DATASET_REGISTRY below), never this constant.
_NUM_CLASSES = 10

_model_cache: Dict[Tuple[str, int], nn.Module] = {}


def register_model(model_id: str, builder: Callable[[int], nn.Module]) -> None:
    """Registers a custom architecture under `model_id` at runtime, so
    `--model-id model_id` builds it exactly like one of the three
    built-ins -- `builder(num_classes) -> nn.Module`, same contract every
    `_build_*` function above follows. The "module.path:function_name"
    spec `_resolve_model_builder` also accepts below is the zero-code-change
    alternative to calling this directly."""
    _MODEL_REGISTRY[model_id] = builder


def _import_callable(spec: str, *, kind: str) -> Callable:
    """Resolves a `"module.path:attr_name"` spec to the callable it names
    -- shared by `_resolve_model_builder`/`_resolve_dataset_loader` below,
    the two places this project lets a caller plug in their own code
    without editing this file (docs/coding_guidelines.md 5.C). `kind` ("model"/"dataset")
    only shapes the error message."""
    module_path, sep, attr_name = spec.rpartition(":")
    if not sep or not module_path or not attr_name:
        raise ValueError(f"Malformed {kind} spec {spec!r}; expected 'module.path:function_name'")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ValueError(f"Could not import module {module_path!r} for {kind} spec {spec!r}: {exc}") from exc
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(f"Module {module_path!r} has no attribute {attr_name!r} for {kind} spec {spec!r}") from exc


def _resolve_model_builder(model_id: str) -> Callable[[int], nn.Module]:
    """`model_id` is either a name already in `_MODEL_REGISTRY` (the three
    built-ins, or anything `register_model` added), or a
    `"module.path:function_name"` import spec resolved via
    `_import_callable` -- a caller's own architecture needs zero changes
    to this repo to plug in."""
    if model_id in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[model_id]
    if ":" in model_id:
        return _import_callable(model_id, kind="model")
    raise ValueError(
        f"No QAT backend registered for model_id={model_id!r}; supported built-ins: "
        f"{list(_MODEL_REGISTRY)}, or pass a 'module.path:function_name' custom spec"
    )


def _get_base_model(model_id: str, num_classes: int = _NUM_CLASSES) -> nn.Module:
    """Just the model architecture (no dataset) -- cached separately from
    `get_qat_backend` so `get_layer_groups` (which @planner calls to learn
    real stage names) never has to pay a dataset's download/load cost.
    Cached per `(model_id, num_classes)` since a different num_classes is
    a genuinely different classifier-head shape, not an interchangeable
    instance."""
    cache_key = (model_id, num_classes)
    if cache_key not in _model_cache:
        _model_cache[cache_key] = _resolve_model_builder(model_id)(num_classes)
    return _model_cache[cache_key]


def get_layer_groups(model_id: str, num_classes: int = _NUM_CLASSES) -> Dict[str, List[str]]:
    """Real per-stage layer names for `model_id`, for @planner to propose
    per-stage `layer_configs` against (nodes/planner.py). `num_classes`
    only affects the throwaway instance's head shape, never the stage
    names this returns -- the default is fine for callers (like @planner)
    that only need structure, before any real dataset has been chosen."""
    return group_quantizable_layers(_get_base_model(model_id, num_classes))


def _weight_matrix_shape(module: nn.Module) -> Optional[Tuple[int, int, int]]:
    """`(rows, cols, groups)` of the weight matrix `module` maps onto a
    crossbar, or None if it isn't a mappable layer.

    `rows` is the input-vector length one crossbar pass consumes and `cols`
    the number of output channels it drives, following the standard
    im2col/GEMM view of a conv every CIM mapping paper uses: an
    `in_ch x k x k` patch is flattened down one column of rows, and each
    output channel gets its own crossbar column. `groups` is `Conv2d.groups`
    -- 1 for an ordinary dense/pointwise conv, `in_channels` for a depthwise
    conv, whose weight matrix is block-diagonal rather than dense (see
    `tools.cim_physics.crossbar_passes` for why that distinction changes the
    real cost by an order of magnitude). `rows` is deliberately the
    *per-group* input length (`in_channels // groups * k * k`), so a
    depthwise layer reports the small block it actually computes, not the
    dense matrix it would be if the zeros were stored."""
    if isinstance(module, nn.Conv2d):
        kernel_h, kernel_w = module.kernel_size
        rows = (module.in_channels // module.groups) * kernel_h * kernel_w
        return rows, module.out_channels, module.groups
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features, 1
    return None


def get_stage_weight_shapes(
    model_id: str, num_classes: int = _NUM_CLASSES
) -> Dict[str, List[Tuple[int, int, int]]]:
    """Per-stage real weight-matrix shapes, keyed by the same stage names
    `get_layer_groups` returns -- what `tools.cim_physics.simulate_cim_profile`
    needs to charge a layer its *real* crossbar cost instead of assuming
    every stage is exactly one full `crossbar_rows x crossbar_cols` array.

    Each stage maps to the list of `(rows, cols, groups)` triples of the
    real layers grouped under it, in `group_quantizable_layers` order. A
    stage is a *group* of layers (resnet18's `layer4.1` alone holds two
    3x3x512x512 convs), which is exactly why charging one array per stage
    understated real cost by ~70x -- see that module's docstring.

    Uses the same cached architecture instance as `get_layer_groups`, so
    this costs one dict walk after the first call and never downloads a
    dataset."""
    model = _get_base_model(model_id, num_classes)
    named = dict(model.named_modules())
    shapes: Dict[str, List[Tuple[int, int, int]]] = {}
    for stage, layer_names in group_quantizable_layers(model).items():
        stage_shapes = []
        for layer_name in layer_names:
            shape = _weight_matrix_shape(named.get(layer_name))
            if shape is not None:
                stage_shapes.append(shape)
        shapes[stage] = stage_shapes
    return shapes


# Fine-tuning set (backward pass -- the per-QAT-trial cost driver), the
# early-stopping monitor (forward-only, evaluated once per epoch), and the
# final-report set (forward-only, evaluated exactly once per trial no matter
# how many epochs run). TEST_SIZE can be much larger than TRAIN_SIZE for
# comparatively little extra wall-clock time precisely because it's only
# ever touched once: 1000 images is ~8x CIFAR10's old 128-image test subset,
# which cuts the accuracy figure's binomial-proportion 95% CI from roughly
# +/-8pp down to roughly +/-3pp -- the earlier size was too small to treat
# as more than a rough directional signal between candidates.
#
# These were hardcoded module constants (docs/coding_guidelines.md 5.C: config-driven, not
# hardcoded) -- a deployment validating against its *real* target dataset
# rather than this demo subset had no way to widen scope without editing
# this file. Now env-var overridable (same pattern as llm.py's
# TESSELLATE_LLM_* / TESSELLATE_PLANNER_MODEL), defaulting to the values above so
# nothing changes for a caller that doesn't opt in. `_FULL_SPLIT` lets
# TESSELLATE_QAT_TRAIN_SIZE/TESSELLATE_QAT_TEST_SIZE request "every image CIFAR10's
# own split has" without hand-computing 50000/10000 -- resolved against the
# actual loaded dataset length in `_build_cifar10_loaders`, not guessed here.
# Widening these multiplies real per-trial wall-clock -- much less severely
# on a machine `_resolve_device` finds a GPU on than on CPU (see this
# module's docstring), but still the caller's tradeoff to make, not
# silently defaulted for them.
_TRAIN_SIZE_ENV = "TESSELLATE_QAT_TRAIN_SIZE"
_VAL_SIZE_ENV = "TESSELLATE_QAT_VAL_SIZE"
_TEST_SIZE_ENV = "TESSELLATE_QAT_TEST_SIZE"
_BATCH_SIZE_ENV = "TESSELLATE_QAT_BATCH_SIZE"
_SEED_ENV = "TESSELLATE_QAT_SEED"
_MAX_EPOCHS_ENV = "TESSELLATE_QAT_MAX_EPOCHS"

_DEFAULT_TRAIN_SIZE = 512
_DEFAULT_VAL_SIZE = 128
_DEFAULT_TEST_SIZE = 1000
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_DATASET_SEED = 0
_DEFAULT_MAX_EPOCHS = 5

_FULL_SPLIT = "full"


def _resolve_size_env(env_var: str, default: int, *, allow_full: bool) -> int | str:
    raw = os.environ.get(env_var)
    if not raw:
        return default
    if allow_full and raw.strip().lower() == _FULL_SPLIT:
        return _FULL_SPLIT
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _resolve_int_env(env_var: str, default: int) -> int:
    raw = os.environ.get(env_var)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def resolve_max_epochs() -> int:
    """`TESSELLATE_QAT_MAX_EPOCHS` (same env-var-overridable pattern as
    `TESSELLATE_QAT_TRAIN_SIZE`/etc. above), defaulting to `_DEFAULT_MAX_EPOCHS`
    -- the single source of truth `nodes/tuner.py` and
    `tools/batch_warmup.py` both call instead of each keeping their own
    hardcoded `_DEFAULT_MAX_EPOCHS` constant "in lockstep" (a real
    duplication risk: nothing enforced the two copies actually staying
    equal, and only one of them updating silently would make the
    sequential graph loop and the parallel warm-up batch train real
    candidates for different numbers of epochs without either caller
    knowing). Widening this multiplies every real QAT trial's wall-clock
    linearly -- the same caller's tradeoff to make as `TESSELLATE_QAT_TRAIN_SIZE`,
    not silently defaulted for them."""
    return _resolve_int_env(_MAX_EPOCHS_ENV, _DEFAULT_MAX_EPOCHS)


def _resolve_split_sizes(
    train_size: int | str, val_size: int, test_size: int | str, train_available: int, test_available: int
) -> Tuple[int, int]:
    """Resolves `_FULL_SPLIT` sentinels against the actual split sizes and
    validates the result fits -- pulled out of `_build_cifar10_loaders` so
    it's unit-testable without a real CIFAR10 download (`train_available`/
    `test_available` are just plain ints here, not `len(some_dataset)`).
    Raises `ValueError` (not a silent clamp) on a misconfigured explicit
    size -- a silently truncated split would produce a real but misleading
    accuracy number rather than a loud, immediate failure."""
    resolved_train_size = train_available - val_size if train_size == _FULL_SPLIT else train_size
    resolved_test_size = test_available if test_size == _FULL_SPLIT else test_size

    if resolved_train_size + val_size > train_available:
        raise ValueError(
            f"{_TRAIN_SIZE_ENV} ({resolved_train_size}) + {_VAL_SIZE_ENV} ({val_size}) exceeds CIFAR10's "
            f"{train_available}-image train split -- reduce one of them or set {_TRAIN_SIZE_ENV}={_FULL_SPLIT!r}."
        )
    if resolved_test_size > test_available:
        raise ValueError(
            f"{_TEST_SIZE_ENV} ({resolved_test_size}) exceeds CIFAR10's {test_available}-image test split."
        )
    return resolved_train_size, resolved_test_size


def _build_cifar10_loaders(
    train_size: int | str, val_size: int, test_size: int | str, batch_size: int, seed: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Three *disjoint* loaders: `train`/`val` are non-overlapping slices of
    CIFAR10's own 50k-image train split (so validation-set early stopping
    never sees a training example), and `test` is CIFAR10's separate
    10k-image test split, entirely untouched until the one final-accuracy
    call in `run_qat_tuning`. `train_size`/`test_size` may be the literal
    string `"full"` (`_FULL_SPLIT`) to use every image in that split;
    `train_size="full"` reserves `val_size` images for validation first and
    uses the remainder for training, since both can't literally claim the
    whole 50k-image train split at once."""
    import torchvision.transforms as T
    from torch.utils.data import Subset
    from torchvision.datasets import CIFAR10

    transform = T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            # ImageNet channel stats -- matches the pretrained resnet18's
            # expected input normalization.
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_full = CIFAR10(root=str(_CACHE_ROOT / "cifar10"), train=True, download=True, transform=transform)
    test_full = CIFAR10(root=str(_CACHE_ROOT / "cifar10"), train=False, download=True, transform=transform)

    resolved_train_size, resolved_test_size = _resolve_split_sizes(
        train_size, val_size, test_size, len(train_full), len(test_full)
    )

    generator = torch.Generator().manual_seed(seed)
    train_val_idx = torch.randperm(len(train_full), generator=generator).tolist()
    train_idx = train_val_idx[:resolved_train_size]
    val_idx = train_val_idx[resolved_train_size : resolved_train_size + val_size]
    test_idx = torch.randperm(len(test_full), generator=generator)[:resolved_test_size].tolist()

    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(train_full, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(test_full, test_idx), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


DatasetLoader = Callable[[], Tuple[DataLoader, DataLoader, DataLoader, int]]

_DEFAULT_DATASET_ID = "cifar10"


def _load_cifar10() -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """The built-in `"cifar10"` dataset loader -- same env-var-driven
    sizing as always (`TESSELLATE_QAT_*`, resolved at call time so
    `monkeypatch.setenv` before the first call still takes effect), now
    behind the same zero-argument `DatasetLoader` contract every custom
    dataset (`register_dataset`/a `"module:function"` spec) follows."""
    train_loader, val_loader, test_loader = _build_cifar10_loaders(
        train_size=_resolve_size_env(_TRAIN_SIZE_ENV, _DEFAULT_TRAIN_SIZE, allow_full=True),
        val_size=_resolve_int_env(_VAL_SIZE_ENV, _DEFAULT_VAL_SIZE),
        test_size=_resolve_size_env(_TEST_SIZE_ENV, _DEFAULT_TEST_SIZE, allow_full=True),
        batch_size=_resolve_int_env(_BATCH_SIZE_ENV, _DEFAULT_BATCH_SIZE),
        seed=_resolve_int_env(_SEED_ENV, _DEFAULT_DATASET_SEED),
    )
    return train_loader, val_loader, test_loader, _NUM_CLASSES


_DATASET_REGISTRY: Dict[str, DatasetLoader] = {_DEFAULT_DATASET_ID: _load_cifar10}


def register_dataset(dataset_id: str, loader: DatasetLoader) -> None:
    """Registers a custom dataset loader under `dataset_id` at runtime --
    `loader()` takes no arguments and returns `(train_loader, val_loader,
    test_loader, num_classes)`, the same three-disjoint-split contract
    `_load_cifar10` follows. Lets a deployment point `--dataset-id` at its
    own real target dataset (ImageNet, a proprietary dataset, ...) without
    editing this file -- the dataset's own module owns whatever
    slicing/sizing logic it needs; nothing here assumes CIFAR10's
    env-var-driven sizing is universal. The "module.path:function_name"
    spec `_resolve_dataset_loader` also accepts below is the
    zero-code-change alternative to calling this directly."""
    _DATASET_REGISTRY[dataset_id] = loader


def _resolve_dataset_loader(dataset_id: str) -> DatasetLoader:
    if dataset_id in _DATASET_REGISTRY:
        return _DATASET_REGISTRY[dataset_id]
    if ":" in dataset_id:
        return _import_callable(dataset_id, kind="dataset")
    raise ValueError(
        f"No dataset loader registered for dataset_id={dataset_id!r}; supported built-ins: "
        f"{list(_DATASET_REGISTRY)}, or pass a 'module.path:function_name' custom spec"
    )


_backend_cache: Dict[Tuple[str, str], QATBackend] = {}


def get_qat_backend(model_id: str, dataset_id: Optional[str] = None) -> QATBackend:
    """Real backend, cached per `(model_id, dataset_id)` for the life of the
    process (the ~44MB pretrained-weight download and ~170MB CIFAR10
    download only happen once; both cache to disk under `.cache/` after
    that). Each `run_qat_tuning` call still deep-copies `base_model` before
    mutating, so the cached instance itself is never fine-tuned in place.
    `dataset_id=None` (the default) resolves to `"cifar10"` -- unchanged
    behavior for any existing caller that doesn't pass one."""
    resolved_dataset_id = dataset_id or _DEFAULT_DATASET_ID
    cache_key = (model_id, resolved_dataset_id)
    if cache_key not in _backend_cache:
        train_loader, val_loader, test_loader, num_classes = _resolve_dataset_loader(resolved_dataset_id)()
        base_model = _get_base_model(model_id, num_classes)
        _backend_cache[cache_key] = QATBackend(
            base_model=base_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            layer_groups=group_quantizable_layers(base_model),
        )
    return _backend_cache[cache_key]
