"""Composed Hydra YAMLs validated against a typed, deliberately bounded schema."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import re

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


@dataclass
class Model:
    id: str = "EleutherAI/pythia-14m"
    revision: str = "5b020995bfc7aee2931b0f35bd70cf7ee8b1db62"
    tokenizer_revision: str = "5b020995bfc7aee2931b0f35bd70cf7ee8b1db62"
    checkpoint: str = "step1000"
    n_layers: int = 6
    architecture: str = "parallel"
    cache_dir: str = "models/hf-cache"


@dataclass
class Task:
    name: str = "induction"
    scoring_version: str = "induction-repeat-target-logprob-v1"
    examples: int = 4
    pattern_length: int = 8
    seed: int = 42
    calibration_examples: int = 16
    source_directory: str = "artifacts/tigges-2024-source"


@dataclass
class Replacement:
    name: str = "zero"
    estimator: str = "exact"
    direction: str = "noising"
    donor_policy: str = "matched"
    donor_seed: int = 43
    recovery_epsilon: float = 1e-6
    mean_reduction: str = "position"


@dataclass
class ComponentSelection:
    hook: str = "hook_embed"
    head: Optional[int] = None


@dataclass
class Scope:
    resolution: str = "mixed"
    positions: str = "all"
    components: List[ComponentSelection] = field(default_factory=list)


@dataclass
class Graph:
    version: str = "influence-v2"
    resolution: str = "component"
    similarity_threshold: float = .7
    epsilon: float = 1e-8


@dataclass
class Resources:
    device: str = "cpu"
    dtype: str = "float32"
    disable_optional_vision: bool = False


@dataclass
class Execution:
    resume: bool = False
    backend: str = "serial"
    example_batch_size: int = 4
    intervention_batch_size: int = 4
    examples_per_shard: int = 4


@dataclass
class Logging:
    root: str = "logs"
    rotation_bytes: int = 10485760
    telemetry_interval_seconds: float = 60.
    retain_all_attempts: bool = True
    console_level: str = "INFO"
    file_level: str = "INFO"
    heartbeat_interval_seconds: float = 10.
    progress_interval_seconds: float = 10.
    timezone: str = "Europe/Rome"


@dataclass
class Export:
    inputs: List[str] = field(default_factory=list)
    thresholds: List[float] = field(default_factory=lambda: [.1, .3, .5, .7])
    quantile: float = .95
    include_activations: bool = False


@dataclass
class Sweep:
    """Checkpoint-grouped grid; each checkpoint is loaded once for all of its cells."""
    checkpoints: List[str] = field(default_factory=list)
    tasks: List[str] = field(default_factory=list)
    replacements: List[str] = field(default_factory=list)
    output_root: str = "runs/sweep"
    examples: Optional[int] = None
    calibration_examples: Optional[int] = None
    skip_incompatible: bool = True
    evict_checkpoint_cache: bool = False
    # none | cosine16 (keep per-example cosine as float16) | means (keep only mean similarities)
    compact: str = "none"


@dataclass
class Experiment:
    command: str = "inspect"
    model: Model = field(default_factory=Model)
    task: Task = field(default_factory=Task)
    replacement: Replacement = field(default_factory=Replacement)
    scope: Scope = field(default_factory=Scope)
    graph: Graph = field(default_factory=Graph)
    resources: Resources = field(default_factory=Resources)
    export: Export = field(default_factory=Export)
    execution: Execution = field(default_factory=Execution)
    logging: Logging = field(default_factory=Logging)
    sweep: Sweep = field(default_factory=Sweep)
    input_matrix: str = "runs/induction-development/similarities.csv"
    output_directory: str = "runs/development"


PYTHIA = re.compile(r"EleutherAI/pythia-(14m|31m|70m|160m|410m|1b|1\.4b|2\.8b|6\.9b|12b)(-deduped)?(-seed[1-9])?")


def validate(config: Experiment):
    if config.command not in {"inspect", "analyze", "run", "export", "sweep"}:
        raise ValueError("command must be inspect, analyze, run, export or sweep")
    if config.replacement.name not in {"zero", "mean", "donor"} or config.replacement.estimator != "exact":
        raise ValueError("Replacement must be exact zero, mean or donor")
    if config.replacement.direction not in {"noising", "denoising"} or config.replacement.donor_policy not in {"matched", "resample"}:
        raise ValueError("Invalid replacement direction or donor policy")
    if config.replacement.mean_reduction not in {"position", "global"}:
        raise ValueError("mean_reduction must be position (aligned example mean) or global")
    if config.replacement.direction == "denoising" and (config.replacement.name, config.replacement.donor_policy) != ("donor", "matched"):
        raise ValueError("Denoising (clean-to-corrupt recovery) requires matched clean donors")
    if config.replacement.name == "donor" and config.replacement.donor_policy == "matched" and config.task.name == "induction":
        raise ValueError("Matched donors require a paired task; use donor_policy=resample for induction")
    if config.replacement.recovery_epsilon <= 0 or config.task.calibration_examples < 0:
        raise ValueError("Invalid calibration/recovery configuration")
    if config.replacement.name == "mean" and config.task.calibration_examples < 1:
        raise ValueError("Mean replacement requires a disjoint calibration split")
    if config.scope.positions != "all" or config.scope.resolution not in {"component", "head", "mixed"}:
        raise ValueError("Scopes must select existing components/heads across all positions")
    if not PYTHIA.fullmatch(config.model.id):
        raise ValueError("Only validated Pythia (GPT-NeoX parallel-block) adapters are implemented")
    if config.model.n_layers < 1:
        raise ValueError("model.n_layers must be positive")
    if config.command == "sweep":
        sweep = config.sweep
        if not sweep.checkpoints or not sweep.tasks or not sweep.replacements:
            raise ValueError("A sweep needs checkpoints, tasks and replacements")
        if len(set(sweep.checkpoints)) != len(sweep.checkpoints) or any(not re.fullmatch(r"step\d+", c) for c in sweep.checkpoints):
            raise ValueError("Sweep checkpoints must be unique numeric step labels")
        if any(not re.fullmatch(r"[a-z0-9_]+", name) for name in [*sweep.tasks, *sweep.replacements]):
            raise ValueError("Sweep tasks/replacements name YAMLs under configs/task and configs/replacement")
        if sweep.compact not in {"none", "cosine16", "means"}:
            raise ValueError("sweep.compact must be none, cosine16 or means")
        if (sweep.examples is not None and sweep.examples < 1) or (sweep.calibration_examples is not None and sweep.calibration_examples < 0):
            raise ValueError("Invalid sweep example counts")
    elif not re.fullmatch(r"[0-9a-f]{40}", config.model.revision) and not (config.command in {"run", "inspect"} and config.model.revision == "resolve"):
        raise ValueError("model.revision must be an immutable 40-character SHA (or resolve, for run/inspect)")
    if not re.fullmatch(r"step\d+", config.model.checkpoint):
        raise ValueError("model.checkpoint must be a numeric step label")
    versions = {"induction": "induction-repeat-target-logprob-v1"}
    from .tasks.paper import VERSIONS
    versions.update(VERSIONS)
    if versions.get(config.task.name) != config.task.scoring_version:
        raise ValueError("Unknown task/scoring contract")
    if config.task.examples < 1 or config.task.pattern_length < 2:
        raise ValueError("Task example count must be positive and induction patterns at least two tokens")
    if config.command not in {"sweep"} and not re.fullmatch(r"[0-9a-f]{40}", config.model.tokenizer_revision) and config.model.tokenizer_revision != "resolve":
        raise ValueError("tokenizer_revision must be an immutable SHA")
    for level in (config.logging.console_level, config.logging.file_level):
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("Invalid logging level")
    import math
    for interval in (config.logging.heartbeat_interval_seconds, config.logging.progress_interval_seconds):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Logging intervals must be finite and positive")
    if config.logging.rotation_bytes < 1024 or config.logging.telemetry_interval_seconds < 0 or not math.isfinite(config.logging.telemetry_interval_seconds):
        raise ValueError("Invalid log rotation/telemetry settings")
    if not config.logging.retain_all_attempts:
        raise ValueError("Automatic log deletion is not supported")
    from zoneinfo import ZoneInfo
    ZoneInfo(config.logging.timezone)
    if config.resources.device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda; select GPU indices through CUDA_VISIBLE_DEVICES")
    if config.resources.dtype not in {"float32", "bfloat16"}:
        raise ValueError("Supported precision: float32 or bfloat16 with FP32 metric reductions")
    if config.execution.backend not in {"serial", "optimized"} or min(config.execution.example_batch_size, config.execution.intervention_batch_size, config.execution.examples_per_shard) < 1:
        raise ValueError("Invalid execution backend or batch sizes")
    if config.execution.backend == "optimized" and config.execution.examples_per_shard % config.execution.example_batch_size:
        raise ValueError("examples_per_shard must be a multiple of example_batch_size for exact resume")
    if config.model.architecture not in {"parallel", "sequential"}:
        raise ValueError("Unknown architecture ordering")
    if config.command in {"run", "sweep"} and config.model.architecture != "parallel":
        raise ValueError("Pythia reference requires parallel blocks")
    if config.graph.version != "influence-v2":
        raise ValueError("Unknown graph version")
    if config.graph.resolution not in {"component", "head", "mixed"}:
        raise ValueError("Unknown graph resolution")
    if config.graph.version == "influence-v2" and config.graph.resolution == "mixed":
        raise ValueError("Corrected graphs require non-overlapping component/head resolution")
    if not -1 <= config.graph.similarity_threshold <= 1 or not 0 < config.graph.epsilon < float("inf"):
        raise ValueError("Invalid similarity threshold or distance epsilon")
    if not 0 <= config.export.quantile <= 1 or any(not -1 <= value <= 1 for value in config.export.thresholds):
        raise ValueError("Invalid export quantile/threshold")
    if not config.output_directory:
        raise ValueError("output_directory is required")
    return config


def load(config_dir, name="config", overrides=()):
    with initialize_config_dir(config_dir=str(Path(config_dir).resolve()), version_base="1.3"):
        composed = compose(config_name=name, overrides=list(overrides))
    schema = OmegaConf.structured(Experiment)
    resolved = OmegaConf.merge(schema, composed)
    OmegaConf.resolve(resolved)
    return validate(OmegaConf.to_object(resolved)), resolved
