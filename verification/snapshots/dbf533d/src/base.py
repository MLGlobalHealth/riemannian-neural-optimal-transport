from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    network_type: str = "mlp"  # "mlp" or "residual"
    hidden_dims: Tuple[int, ...] = (128, 128)
    n_landmarks: int = 128
    use_layernorm: bool = True
    last_scale: float = 0.01
    activation: str = "silu"  # "silu", "hard_swish", "leaky_relu", "gelu", "selu", "softplus"
    leaky_slope: float = 0.01
    softplus_beta: float = 1
    max_dist: float = 3.14159  # π for S2, used by ResidualMLP

@dataclass
class SolverConfig:
    """Argmin solver configuration"""
    inner_steps: int = 2500
    inner_lr: float = 5e-2
    grad_clip: Optional[float] = None
    lr_decay: bool = False
    tolerance: float = 1e-6
    min_steps: int = 1000
    momentum: float = 0.0  # Riemannian Heavy Ball momentum (ignored if use_adam=True)
    logsumexp_init: bool = True  # If True, warm-start from soft-argmin over target samples
    logsumexp_gamma: float = 0.1  # Temperature for soft-argmin (smaller = harder min)
    use_adam: bool = True  # Use Adam optimizer (per-coordinate adaptive LR)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

@dataclass
class TrainingConfig:
    """Training loop configuration"""
    n_steps: int = 1000
    batch_size: int = 256
    learning_rate: float = 1e-3
    lr_decay: bool = False  # If True, use cosine LR decay
    lr_decay_alpha: float = 0.05  # Final LR = learning_rate * alpha
    log_every: int = 1
    eval_every: int = None
    eval_size: int = 1024
    seed: int = 12345

@dataclass
class ExperimentConfig:
    """Full experiment configuration"""
    manifold_name: str = "S2"
    base_density: str = "SphereUniform"
    target_density: str = "SphereWrappedNormal"
    jax_platform: Optional[str] = "gpu"

    model: ModelConfig = field(default_factory=ModelConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)