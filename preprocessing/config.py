"""
Unified configuration for the AirfRANS preprocessing pipeline.

Covers both steps:
  1. Downsampling  (downsample_airfrans.py)
  2. Edge building (edges_from_downsampled.py)
"""
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class DownsampleConfig:
    """Arguments for adaptive-voxel downsampling (step 1)."""
    root: str = "Dataset"
    task: Literal["scarce", "full"] = "scarce"
    out_dir: str = "downsampled_graphs"
    limit_train: Optional[int] = None
    limit_test: Optional[int] = None
    target_min_nodes: int = 15_000
    target_max_nodes: int = 30_000
    voxel_frac: float = 0.01
    voxel_iters: int = 5


@dataclass
class EdgeConfig:
    """Arguments for edge construction (step 2)."""
    in_dir: str = "downsampled_graphs"
    out_dir: str = "prebuilt_edges"
    task: Literal["scarce", "full"] = "scarce"
    global_radius: float = 0.02
    surface_radius: float = 0.01
    max_num_neighbors: int = 48
    surface_ring: bool = True
    denormalize: bool = False
    min_degree: int = 2
    knn_backup_k: int = 4
    knn_max_radius: float = 0.05


@dataclass
class PreprocessingConfig:
    """Full preprocessing pipeline configuration (step 1 + step 2)."""
    downsample: DownsampleConfig = field(default_factory=DownsampleConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
