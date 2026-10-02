import os
import sys
from pathlib import Path
from typing import Tuple, Union, Optional


def get_project_root() -> Path:
    """
    Finds and returns the absolute Path to the water_segmentation project root.
    Works whether invoked from:
      - <root> (e:/Cellula_Technologies/Project2)
      - <root>/water_segmentation
      - <root>/water_segmentation/notebooks
      - <root>/water_segmentation/tests
    """
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent,
        Path("e:/Cellula_Technologies/Project2/water_segmentation"),
        Path("e:/Cellula_Technologies/Project2"),
    ]
    
    for candidate in candidates:
        if (candidate / "data" / "raw" / "images").exists() and (candidate / "src").exists():
            return candidate.resolve()
        if (candidate / "water_segmentation" / "data" / "raw" / "images").exists():
            return (candidate / "water_segmentation").resolve()
            
    # Default fallback
    fallback = Path("e:/Cellula_Technologies/Project2/water_segmentation").resolve()
    return fallback


def get_data_dirs() -> Tuple[Path, Path, Path, Path, Path]:
    """
    Returns (img_dir, lbl_dir, splits_dir, stats_dir, checkpoints_dir)
    guaranteed to exist and resolve correctly from any CWD.
    """
    root = get_project_root()
    img_dir = root / "data" / "raw" / "images"
    lbl_dir = root / "data" / "raw" / "labels"
    splits_dir = root / "data" / "splits"
    stats_dir = root / "data" / "statistics"
    checkpoints_dir = root / "checkpoints"
    
    return img_dir, lbl_dir, splits_dir, stats_dir, checkpoints_dir


def resolve_file(file_path: Union[str, Path]) -> Path:
    """
    Resolves a file path that might be given relative to project root or cwd.
    """
    p = Path(file_path)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
        
    root = get_project_root()
    # Check relative to root
    if (root / p).exists():
        return (root / p).resolve()
    # Check removing leading water_segmentation if root already ends with water_segmentation
    parts = p.parts
    if parts and parts[0] == "water_segmentation":
        sub_p = Path(*parts[1:])
        if (root / sub_p).exists():
            return (root / sub_p).resolve()
            
    # Check parent
    if (Path.cwd().parent / p).exists():
        return (Path.cwd().parent / p).resolve()
        
    return p.resolve()
