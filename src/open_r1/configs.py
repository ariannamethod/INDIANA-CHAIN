from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DatasetConfig:
    id: str
    config: Optional[str] = None
    split: str = "train"
    columns: Optional[list[str]] = None
    weight: Optional[float] = None


@dataclass
class DatasetMixtureConfig:
    datasets: list[DatasetConfig]
    seed: int = 0
    test_split_size: Optional[float] = None


@dataclass
class ScriptArguments:
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_mixture: Optional[dict[str, Any]] = None


@dataclass
class GRPOConfig:
    pass


@dataclass
class SFTConfig:
    pass


@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(default_factory=list)
