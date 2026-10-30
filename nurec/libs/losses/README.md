<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Losses Module

## Overview

The Losses module computes loss functions during neural rendering training. The fused kernels live in gsplat; this repository shapes its tensors for them, wraps them as modules, and aggregates the results under a configuration-driven selection. It separates concerns across dispatch, module wrapping and aggregation.

## Architecture

The Losses module follows a four-layer architecture pattern:

1. **Layer 0: Fused loss kernels** - gsplat's fused CUDA kernels compute the loss types. They are not
   implemented in this repository.
2. **Layer 1: Dispatcher** - `CudaLossesFunction` shapes this repository's tensors and factors for those
   kernels and calls gsplat's functional entry points. It is a plain callable, not a
   `torch.autograd.Function`: gsplat's dispatches build their own graph and carry their own backward.
3. **Layer 2: Loss Modules** - PyTorch nn.Module(s) that manage loss lifecycle and data preparation, one
   per loss type (e.g. `RGBLoss`, `LidarLoss`, `BackgroundLoss`). `ModuleLosses` covers the names it
   lists in `available` through the fused path; the rest are pure PyTorch in `render_losses.py`.
4. **Layer 3: LossAggregator** - Configuration-driven orchestrator that aggregates and coordinates all losses

Each layer only accesses the immediate layer below it: Layer 3 → Layer 2 → Layer 1 → Layer 0. This is not a hard requirement of the design, just a nice-to-have for the losses module. The layered design ensures clear separation of concerns and maintainability, and the only requirement is that any given layer only depends on itself or layers below it, i.e. layer N depends on layers M where M ≤ N.

---

### Layer 0: Fused loss kernels

**Location:** gsplat, `gsplat/losses/`.

The fused kernels and their gradients are implemented and tested in gsplat. This
repository does not carry a loss kernel, a Slang source file, or an autograd
function for one.

### Layer 1: `CudaLossesFunction` (Dispatcher)

**Location:** `nurec/libs/losses/functional/cuda_losses_function.py`

Shapes this repository's tensors and per-loss factors for gsplat's fused modules
and calls them, then unpacks the returned maps and scalars. It is a plain
callable: gsplat's dispatches build their own graph and carry their own backward,
so there is nothing here for `torch.autograd.Function` to do.

Input contracts (dtype, shape and contiguity) are asserted before dispatch.

### Layer 2: Loss Modules (Neural Network Modules)

**Conceptual Description:** Layer 2 represents a set of `torch.nn.Module` instances, conceptually one per loss type (RGB loss, LiDAR loss, background loss, etc.). Each module manages the lifecycle of its loss computation, including data preparation, loss selection, and result packaging.

**Implementation Strategies:**

This layer can be implemented in two ways:

1. **CUDA Fused Implementation (Current Location: `nurec/libs/losses/models/losses_module.py`):**

   - A single stateful `torch.nn.Module` class called `ModuleLosses` that fuses multiple loss computations
   - Manages all CUDA-based losses together for performance optimization
   - Invokes the fused CUDA kernel (Layer 0) via the dispatcher (Layer 1)
   - Handles data preparation for all losses simultaneously

2. **PyTorch Separate Implementation (Current Location: `nurec/libs/losses/models/render_losses.py`, `nurec/libs/losses/models/primitive_losses.py`):**
   - Multiple separate `torch.nn.Module` classes, one per loss type (e.g., `RGBLoss`, `LidarLoss`, `BackgroundLoss`)
   - Each class inherits from `BaseRenderLoss` or `BasePrimitiveLoss`
   - Each loss is computed independently with its own kernel invocation
   - Provides flexibility but with more kernel launch overhead

The CUDA fused implementation is preferred for performance, while the PyTorch separate implementation is used for losses not on the fused path or requiring specialized computation.

---

#### Layer 2A: ModuleLosses (Fused CUDA Implementation)

**Current location:** `nurec/libs/losses/models/losses_module.py`

**Description:** A stateful `torch.nn.Module` that manages the lifecycle of multiple CUDA-based losses simultaneously. This implementation fuses data preparation and kernel invocation for all configured fused losses.

**Key Responsibilities:**

- **Loss Registration:** Maintains a list of available CUDA-based loss types
- **Configuration Management:** Tracks which losses are enabled and their lambda weights
- **Data Preparation:** Extracts and reshapes tensors from model outputs and target data
- **Loss Selection:** Determines which losses to compute based on step, configuration, and data availability
- **Result Packaging:** Wraps computed losses into `LossReturn` objects with lambda weights and reduction functions

**Initialization API Design:**

```python
def __init__(self) -> None:
    """
    Initialization API design:

    Responsibilities:
    - Initialize available loss types list
    - Create sum reduction function for fused losses
    - Pre-allocate dummy tensors for unused loss types
      - Shape: minimum required dimensions (typically [1, 1, 1, ...])
      - Dtype: matches actual tensor types (int32 for flags, float32 for others)

    Loss Instances:
    - self.losses contains SlangBaseLoss instances
    - SlangBaseLoss is a concrete class inheriting from BaseLoss
    - Provides configuration functionalities such as step-based execution control
    """
    super().__init__()
    self.available: list[str] = [...]  # Example: ["rgb_l1_mean", "lidar_l1_mean", ...]
    self.losses: list[SlangBaseLoss] = []
    self.sum_reduce_fn: SumReduceFn = SumReduceFn(...)
    # Dummy tensors for each loss type category
    self.dummy_flags: torch.Tensor = ...
    self.dummy_pred: torch.Tensor = ...
    self.dummy_gt: torch.Tensor = ...
```

**Forward API Design:**

```python
def forward(
    self,
    step: int,
    model: BaseModel,
    results: GaussiansCompositeReturn,
    target: DataAndRenderingBatch,
) -> dict[str, LossReturn]:
    """
    Forward pass API design:

    Inputs:
    - step: Current training step (for step-based loss scheduling)
    - model: Model instance (for accessing model parameters, post-processing modules)
    - results: Rendered outputs containing predictions
    - target: Ground-truth data and labels

    Returns:
    - dict[str, LossReturn]: Dictionary mapping loss names to LossReturn objects

    Responsibilities:
    1. Determine which losses should run:
       - Check loss.should_run_fn(step) for step-based execution control
       - Check is_enabled() for data availability
    2. Extract and prepare tensors:
       - Load shared data (flags, ground-truth) once
       - Extract loss-specific predictions from results
       - Extract model parameters (e.g., bilateral grids)
       - Reshape tensors to match kernel expectations
       - Compute normalization factors (1/N_valid)
    3. Use dummy tensors for disabled losses:
       - Replace disabled loss tensors with dummy tensors
       - Set corresponding factors to negative values
    4. Invoke CudaLossesFunction.forward() with prepared tensors (Layer 1)
    5. Package results:
       - Map kernel outputs to loss names
       - Create LossReturn objects with lambda weights
       - Apply reduction functions
    """
```

**Loss Selection and Data Availability:**

**Step-Based Execution Control:**

- Each loss configuration can specify a `start_step` parameter
- Losses only execute when `current_step >= start_step`
- This allows staged training where losses are introduced gradually
- Lambda schedulers can further control loss weighting over time

**Data Availability Checks (`is_enabled`):**

- Checks if required data is available in the input dataset
- **Example:** A LiDAR-based loss configured in the config may be disabled if the input dataset does not contain LiDAR data
- **Example:** A background loss may be disabled if camera has any harmonized label
- Prevents runtime errors from missing data
- Allows configuration to include losses that may not be supported by all datasets

**Data Reuse in Layer 2:**

- When multiple losses share data (e.g., flags), the data is loaded once and reused
- **Example:** Background losses reuse RGB and LiDAR flags tensors rather than loading them separately
- Care must be taken to avoid duplicate loading of shared data
- Shared data extraction happens once per forward pass, then passed to multiple losses

**Example Loss Types:**

Examples of loss types that can be registered in `ModuleLosses`:

- `rgb_l1_mean`: RGB L1 loss with mean reduction
- `lidar_l1_mean`: LiDAR L1 loss with mean reduction
- `background_mse_mean`: Background MSE loss with mean reduction
- `background_lidar_mse_mean`: Background LiDAR MSE loss with mean reduction
- `bilateral_grid_drift_identity_distance_mean`: Bilateral grid drift loss with mean reduction
- `distance_l1_mean`: Distance prediction L1 loss with mean reduction
- `semantic_cross_entropy_mean`: Semantic segmentation cross-entropy loss with mean reduction

---

#### Layer 2B: PyTorch Loss Modules (Separate PyTorch Implementation)

**Current location:** `nurec/libs/losses/models/render_losses.py`, `nurec/libs/losses/models/primitive_losses.py`

**Description:** Multiple separate `torch.nn.Module` classes that implement individual loss computations in pure PyTorch. Each loss type has its own class that inherits from either `BaseRenderLoss` or `BasePrimitiveLoss`.

**Key Characteristics:**

- **Separate Classes:** Each loss type is implemented as an independent class (e.g., `RGBLoss`, `LidarLoss`, `BackgroundLoss`, `DistanceLoss`, `SemanticLoss`)
- **Individual Execution:** Each loss executes independently with its own kernel invocations
- **Flexibility:** Easier to implement specialized or experimental losses without modifying the fused kernel

**Base Class Hierarchy:**

```python
BaseLoss (abstract)
├── BaseRenderLoss (abstract) - Losses applied to rendered results
│   ├── RGBLoss
│   ├── LidarLoss
│   ├── BackgroundLoss
│   ├── DistanceLoss
│   ├── SemanticLoss
│   ├── LPIPSLoss
│   ├── SSIMLoss
│   └── ... (other render losses)
│
└── BasePrimitiveLoss (abstract) - Losses applied to instant-nurec primitives
    ├── PrimitiveDistanceLoss
    ├── PrimitiveSkyDistanceLoss
    ├── PrimitiveSkyMaskLoss
    └── ... (other primitive losses)
```

**Example Loss Classes:**

- `RGBLoss`: Computes loss between rendered RGB and ground-truth RGB
- `LidarLoss`: Computes loss between rendered LiDAR distance and ground-truth distance
- `BackgroundLoss`: Regularization loss for opacity predictions
- `BilateralGridDriftLoss`: Regularization for bilateral grid parameters
- `SemanticLoss`: Cross-entropy loss for semantic segmentation
- `LPIPSLoss`: Perceptual loss using LPIPS network
- `SSIMLoss`: Structural similarity loss
- `PrimitiveDistanceLoss`: Loss for instant-nurec primitive distance supervision

### Layer 3: LossAggregator (Orchestration)

**Current location:** `nurec/libs/losses/orchestration/loss_aggregator.py`

**Description:** The top-level orchestrator that manages all loss functions (both CUDA-fused and PyTorch-separate implementations) according to external configuration. This layer provides a unified interface for loss computation and handles the coordination between different Layer 2 implementations (CUDA fused vs. PyTorch separate).

**Key Responsibilities:**

- **Loss Registration:** Parses OmegaConf configuration to instantiate loss functions
- **Loss Routing:** Routes losses to CUDA fused implementation (Layer 2A) when available, otherwise uses PyTorch separate implementation (Layer 2B)
- **Unified Execution:** Provides a single `__call__` interface for all loss computations regardless of implementation
- **Result Aggregation:** Combines results from both Layer 2A (CUDA fused) and Layer 2B (PyTorch separate) implementations into a unified return structure

**Initialization API Design:**

```python
def __init__(
    self,
    config: LossConfig,
    trainer_config: TrainerConfig,
    **kwargs
) -> None:
    """
    Initialization API design:

    Inputs:
    - config: OmegaConf configuration containing loss definitions
    - trainer_config: Trainer configuration for step-based scheduling
    - kwargs: Optional overrides (e.g., force_disable_cuda for testing)

    Responsibilities:
    1. Parse configuration:
       - Iterate over loss definitions in config
       - Extract loss name, function type, reduction type
       - Construct full_name: "{loss_name}_{loss_fn}_{reduce_name}"
    2. Route losses:
       - Check if full_name is in ModuleLosses.available (Layer 2A)
       - Route to CUDA fused implementation if available, otherwise PyTorch separate implementation
    3. Instantiate losses:
       - Create SlangBaseLoss instances for CUDA fused losses (Layer 2A)
       - Create BaseRenderLoss or BasePrimitiveLoss instances for PyTorch separate losses (Layer 2B)
    """
```

**Execution API Design:**

```python
def __call__(
    self,
    *,
    step: int,
    model: BaseModel,
    results: GaussiansCompositeReturn | None = None,
    target: DataAndRenderingBatch | None = None,
    primitive: BaseInstantNuRecPrimitive | None = None,
    supervision_pack: BaseInstantNuRecSupervisionPack | None = None,
    context: DataAndRenderingBatch | None = None,
) -> LossAggregatorReturn:
    """
    Execution API design:

    Inputs:
    - step: Current training step
    - model: Model instance
    - results: Rendered outputs (for BaseRenderLoss)
    - target: Ground-truth data (for BaseRenderLoss)
    - primitive: instant-nurec primitive (for BasePrimitiveLoss)
    - supervision_pack: instant-nurec supervision pack (for BasePrimitiveLoss)
    - context: Context batch (for BasePrimitiveLoss)

    Returns:
    - LossAggregatorReturn: Aggregated loss results

    Responsibilities:
    1. Execute CUDA fused losses (Layer 2A):
       - Call ModuleLosses.forward() once for all configured fused losses
       - Returns dictionary of LossReturn objects
    2. Execute PyTorch separate losses (Layer 2B - individual kernels):
       - Iterate over PyTorch loss modules
       - Check loss.should_run_fn(step) for each
       - Call loss.forward() individually
       - Collect LossReturn objects
    3. Aggregate results:
       - Combine results from Layer 2A (CUDA fused) and Layer 2B (PyTorch separate)
       - Return LossAggregatorReturn with unified interface
    """
```

**Configuration-Driven Design:**

Loss selection and weighting are controlled entirely through OmegaConf configuration files. The aggregator automatically routes losses to the appropriate implementation based on availability and naming convention (`{loss_name}_{loss_fn}_{reduce_fn}`).

**Loss Types Supported:**

- **BaseRenderLoss:** Losses applied to rendered results (RGB, LiDAR, background, etc.)
- **BasePrimitiveLoss:** Losses applied to instant-nurec primitives (requires primitive, context, supervision_pack)

---

## Design Principles

### 1. Fused Kernel Execution

The CUDA implementation fuses loss computations, improving performance through:

- Reduced kernel launch overhead
- Better memory bandwidth utilization
- Shared data loading across loss types

### 2. Conditional Execution

Loss types are conditionally executed based on:

- Training step (via `start_step` and lambda schedulers)
- Data availability (camera, LiDAR, bilateral grids)
- Configuration (loss enabled/disabled)

### 3. Unified Interface

The `LossAggregator` provides a single entry point for all loss computations, abstracting away the differences between CUDA and Python implementations.

### 4. Configuration-Driven

Loss selection, weighting, and scheduling are controlled through external configuration files, enabling flexible experimentation without code changes.

### 5. Layered Access Pattern

Each layer only accesses the immediate layer below it: Layer 3 → Layer 2 → Layer 1 → Layer 0. This ensures clear separation of concerns and maintainability.

---

## Layer 2 Implementation Notes

**Note:** The losses module supports two implementation strategies at Layer 2:

- **Layer 2A (CUDA Fused):** Preferred implementation for performance-critical losses. Multiple losses are computed in a single fused kernel invocation.
- **Layer 2B (PyTorch Separate):** Used for losses not on the fused path, or losses requiring specialized computation (e.g., LPIPS, SSIM). Each loss has its own kernel invocations.

The SDK design focuses on the CUDA fused architecture (Layer 2A) as the primary implementation path. As losses mature and stabilize, they are migrated from Layer 2B to Layer 2A for improved performance.

---

## Folder Structure

The losses module is organized into a layered folder structure that mirrors the architectural layers:

```
nurec/libs/losses/
├── functional/                      # Layer 1: Dispatcher
│   ├── __init__.py
│   ├── BUILD.bazel
│   └── cuda_losses_function.py     # CudaLossesFunction (plain callable onto gsplat)
│
├── models/                          # Layer 2: Loss Modules
│   ├── __init__.py
│   ├── BUILD.bazel
│   │
│   ├── base_losses.py              # Base classes (BaseLoss, BaseRenderLoss, BasePrimitiveLoss, etc.)
│   ├── registry.py                 # Loss class registration system
│   │
│   ├── losses_module.py      # Layer 2A: ModuleLosses (fused CUDA implementation)
│   ├── render_losses.py            # Layer 2B: PyTorch render losses (RGBLoss, LidarLoss, etc.)
│   ├── primitive_losses.py         # Layer 2B: PyTorch primitive losses
│   │
│   ├── utils.py                     # Utilities used by Layer 2 (_get_bilateral_grids, SSIM helpers, etc.)
│   ├── loss_fns.py                  # Loss function definitions and registry
│   │
│   ├── lambda_schedulers/           # Lambda scheduling submodule
│   │   ├── __init__.py
│   │   ├── BUILD.bazel
│   │   ├── lambda_schedulers.py
│   │   └── registry.py
│   │
│   ├── reduce_functions/            # Reduction functions submodule
│   │   ├── __init__.py
│   │   ├── BUILD.bazel
│   │   ├── reduce_functions.py
│   │   └── registry.py
│   │
│   └── losses_module_test.py        # Tests for loss modules layer
│
├── orchestration/                   # Layer 3: Orchestration
│   ├── __init__.py
│   ├── BUILD.bazel
│   └── loss_aggregator.py          # LossAggregator (orchestrator)
│
├── __init__.py                      # Package initialization
└── BUILD.bazel                      # Root build configuration
```

### Folder Organization Principles

1. **Layer Isolation:** Each layer has its own folder with clear boundaries
2. **Distributed Utilities:** Utilities live in the layer that uses them (higher layers can access lower layer utilities)
3. **Distributed Tests:** Test files live in the layer they test
4. **Submodules in Layer 2:** Lambda schedulers and reduce functions are used primarily by Layer 2 modules, so they reside there
5. **Upward Dependencies Only:** Higher layers can depend on lower layers (Layer 3 → Layer 2 → Layer 1 → Layer 0), but not vice versa

## Integration Points

### With Rendering Pipeline

- Receives `GaussiansCompositeReturn` containing rendered RGB, LiDAR, and other signals
- Extracts opacity and distance predictions from rendering results
- Accesses bilateral grids from model post-processing modules

### With Data Pipeline

- Receives `DataAndRenderingBatch` containing ground-truth labels
- Extracts RGB labels, LiDAR labels, and semantic flags
- Uses ray flags (`RayFlags`) for masking and conditional computation

### With Training Pipeline

- Integrates with `TrainerConfig` for step-based scheduling
- Supports lambda schedulers for dynamic loss weighting
- Returns `LossAggregatorReturn` containing aggregated loss values

---

## Performance Considerations

1. **Memory Efficiency:** Dummy tensors are pre-allocated for unused loss types to avoid dynamic allocation overhead
2. **Contiguous Memory:** All tensors are ensured to be contiguous as required by gsplat before kernel launch (enforced in Layer 1)
3. **Factor-Based Execution:** Negative factors indicate disabled losses, allowing the kernel to skip unnecessary computation
4. **Fused Computation:** Multiple loss types computed in a single kernel reduce synchronization overhead
5. **Data Reuse:** Shared data (flags, ground-truth) is loaded once and reused across multiple losses, avoiding duplicate loading

---

## Extending the Losses Module

New loss implementations and numerical contracts belong in gsplat; the steps below concern their NRE integration.

### Adding New Loss Types to Layer 2A (CUDA Fused Module)

When registering a new loss type in the CUDA fused implementation:

1. **Register Loss Name:**

   - Add loss name to `ModuleLosses.available` list
   - Follow naming convention: `{loss_name}_{loss_fn}_{reduce_fn}`

2. **Add Data Extraction:**

   - Implement data extraction logic for the new loss
   - Reuse shared data extraction when possible
   - Compute normalization factors (1/N_valid)

3. **Update is_enabled:**

   - Add data availability check for the new loss
   - Return False if required data is not available

4. **Add Dummy Tensors:**
   - Pre-allocate dummy tensors for the new loss type
   - Use dummy tensors when loss is disabled

### Adding New Loss Types to Layer 2B (PyTorch Separate Modules)

When implementing a new loss type in PyTorch:

1. **Create Loss Class:**

   - Inherit from `BaseRenderLoss` or `BasePrimitiveLoss`
   - Implement `forward()` to prepare NRE inputs and call the loss implementation
   - Use `@register_loss(name)` decorator for automatic registration

2. **Implement Forward Logic:**

   - Extract required data from inputs (results, target, model)
   - Apply masking and filtering as needed
   - Call `self.apply_loss_fn()` with processed data
   - Return `LossReturn` object

3. **Consider Migration:**
   - If the loss becomes performance-critical, plan migration to Layer 2A (CUDA fused)

### Adding New Loss Types to Layer 3 (Orchestrator)

The orchestrator automatically handles new loss types if:

- Loss name follows naming convention: `{loss_name}_{loss_fn}_{reduce_fn}`
- Loss is registered in `ModuleLosses.available` (for Layer 2A) or via `@register_loss` decorator (for Layer 2B)
- Configuration specifies the loss with proper structure

No changes to Layer 3 are typically needed when adding new loss types.
