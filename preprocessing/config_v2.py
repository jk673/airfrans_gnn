"""
Unified configuration for the AirfRANS preprocessing pipeline (V2).

Changes from V1:
  - QA thresholds for fail-fast validation
  - Voxel representative policy option
"""
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class DownsampleConfigV2:
    """Arguments for adaptive-voxel downsampling (step 1) — V2."""
    root: str = "Dataset"
    task: Literal["scarce", "full"] = "scarce"
    out_dir: str = "Dataset/processed_data/downsampled-graphs"
    limit_train: Optional[int] = None
    limit_test: Optional[int] = None
    target_min_nodes: int = 15_000
    target_max_nodes: int = 30_000
    voxel_frac: float = 0.01
    voxel_iters: int = 5
    # V2: voxel representative selection strategy
    voxel_rep: Literal["gradient", "centroid", "first"] = "gradient"


@dataclass
class EdgeConfigV2:
    """Arguments for edge construction (step 2) — V2."""
    in_dir: str = "Dataset/processed_data/downsampled-graphs"
    out_dir: str = "Dataset/processed_data/prebuilt_edges"
    task: Literal["scarce", "full"] = "scarce"
    global_radius: float = 0.02
    surface_radius: float = 0.01
    max_num_neighbors: int = 48
    surface_ring: bool = True
    denormalize: bool = False
    min_degree: int = 2
    knn_backup_k: int = 4
    knn_max_radius: float = 0.05
    # V2: QA thresholds
    max_isolated_fraction: float = 0.01   # warn if >1% nodes are isolated
    max_low_degree_fraction: float = 0.05  # warn if >5% nodes below min_degree
    qa_fail_fast: bool = False  # raise error instead of warning


@dataclass
class PreprocessingConfigV2:
    """Full preprocessing pipeline configuration (step 1 + step 2) — V2."""
    downsample: DownsampleConfigV2 = field(default_factory=DownsampleConfigV2)
    edge: EdgeConfigV2 = field(default_factory=EdgeConfigV2)
