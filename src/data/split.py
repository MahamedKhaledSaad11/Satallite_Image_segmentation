"""
Group-Aware Leakage-Safe Dataset Splitter.
Splits images into Train, Validation, and Test sets at the DEM elevation group level
to prevent geographic spatial autocorrelation and data leakage.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import tifffile
from PIL import Image

def compute_scene_dem_groups(
    image_dir: str,
    label_dir: str,
    image_ids: List[int],
    bucket_size: int = 10
) -> pd.DataFrame:
    """
    Extracts mean Copernicus DEM elevation and water ratio per image,
    and assigns a discrete elevation group ID.
    """
    img_dir_path = Path(image_dir)
    lbl_dir_path = Path(label_dir)
    
    records = []
    for img_id in image_ids:
        img = tifffile.imread(str(img_dir_path / f"{img_id}.tif"))
        lbl = np.array(Image.open(str(lbl_dir_path / f"{img_id}.png")))
        
        cop_mean = float(img[:, :, 9].mean())
        water_ratio = float((lbl == 1).mean())
        group_id = int(round(cop_mean / bucket_size) * bucket_size)
        
        records.append({
            "image_id": img_id,
            "cop_mean": cop_mean,
            "group_id": group_id,
            "water_ratio": water_ratio
        })
        
    return pd.DataFrame(records)

def create_group_aware_split(
    image_ids: List[int],
    group_ids: List[int],
    water_ratios: List[float],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Splits images into (train_ids, val_ids, test_ids) strictly at group level.
    Uses randomized search to optimize both group capacity and water ratio balance.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    total_images = len(image_ids)
    target_train = int(round(total_images * train_ratio))
    target_val = int(round(total_images * val_ratio))
    target_test = total_images - target_train - target_val
    overall_mean_water = float(np.mean(water_ratios))
    
    # Pre-aggregate group metadata into fast structures
    groups_dict = {}
    for i, g, w in zip(image_ids, group_ids, water_ratios):
        if g not in groups_dict:
            groups_dict[g] = {"ids": [], "water_sum": 0.0}
        groups_dict[g]["ids"].append(i)
        groups_dict[g]["water_sum"] += w

    group_list = []
    for g, data in groups_dict.items():
        sz = len(data["ids"])
        group_list.append({
            "group": g,
            "size": sz,
            "water_sum": data["water_sum"],
            "ids": data["ids"]
        })

    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_split = None
    
    for _ in range(5000):
        indices = rng.permutation(len(group_list))
        
        train_ids, val_ids, test_ids = [], [], []
        train_w_sum, val_w_sum, test_w_sum = 0.0, 0.0, 0.0
        n_train, n_val, n_test = 0, 0, 0
        
        for idx in indices:
            g_data = group_list[idx]
            sz = g_data["size"]
            w_sum = g_data["water_sum"]
            g_ids = g_data["ids"]
            
            can_val = (n_val + sz <= target_val + 4) and (sz <= 30)
            can_test = (n_test + sz <= target_test + 4) and (sz <= 30)
            
            if can_val and (n_val < target_val):
                val_ids.extend(g_ids)
                val_w_sum += w_sum
                n_val += sz
            elif can_test and (n_test < target_test):
                test_ids.extend(g_ids)
                test_w_sum += w_sum
                n_test += sz
            else:
                train_ids.extend(g_ids)
                train_w_sum += w_sum
                n_train += sz
                
        size_penalty = (
            abs(n_train - target_train) * 2.0 +
            abs(n_val - target_val) * 3.0 +
            abs(n_test - target_test) * 3.0
        )
        
        m_train = train_w_sum / max(1, n_train)
        m_val = val_w_sum / max(1, n_val)
        m_test = test_w_sum / max(1, n_test)
        
        water_penalty = (
            abs(m_train - overall_mean_water) * 100.0 +
            abs(m_val - overall_mean_water) * 100.0 +
            abs(m_test - overall_mean_water) * 100.0
        )
        
        total_score = size_penalty + water_penalty
        if total_score < best_score:
            best_score = total_score
            best_split = (train_ids, val_ids, test_ids)
            
    train_ids, val_ids, test_ids = best_split
    
    # Assert mutual exclusivity of groups
    id_to_group = dict(zip(image_ids, group_ids))
    train_grps = {id_to_group[i] for i in train_ids}
    val_grps = {id_to_group[i] for i in val_ids}
    test_grps = {id_to_group[i] for i in test_ids}
    
    assert len(train_grps.intersection(val_grps)) == 0, "Train-Val group leakage!"
    assert len(train_grps.intersection(test_grps)) == 0, "Train-Test group leakage!"
    assert len(val_grps.intersection(test_grps)) == 0, "Val-Test group leakage!"
    
    return train_ids, val_ids, test_ids

def generate_and_save_splits(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    total_count: int = 306,
    bucket_size: int = 10,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Computes groups, performs group-aware split, and writes train.csv, val.csv, test.csv.
    """
    all_ids = list(range(total_count))
    df_meta = compute_scene_dem_groups(image_dir, label_dir, all_ids, bucket_size=bucket_size)
    
    train_ids, val_ids, test_ids = create_group_aware_split(
        image_ids=df_meta["image_id"].tolist(),
        group_ids=df_meta["group_id"].tolist(),
        water_ratios=df_meta["water_ratio"].tolist(),
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed
    )
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    split_dfs = {}
    for name, s_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        sub_df = df_meta[df_meta["image_id"].isin(s_ids)].copy()
        sub_df = sub_df[["image_id", "group_id", "water_ratio"]].sort_values("image_id")
        csv_file = out_path / f"{name}.csv"
        sub_df.to_csv(csv_file, index=False)
        split_dfs[name] = sub_df
        
    return split_dfs

if __name__ == "__main__":
    splits = generate_and_save_splits(
        image_dir="water_segmentation/data/raw/images",
        label_dir="water_segmentation/data/raw/labels",
        output_dir="water_segmentation/data/splits",
        total_count=306,
        bucket_size=10,
        seed=42
    )
    print("=" * 50)
    print("      GROUP-AWARE LEAKAGE-SAFE SPLIT SUMMARY")
    print("=" * 50)
    for name, df in splits.items():
        n_grps = df["group_id"].nunique()
        mean_w = df["water_ratio"].mean()
        print(f"{name.upper():5s}: {len(df):3d} images ({n_grps:2d} groups) | Mean Water Ratio: {mean_w:.3%}")
    print("=" * 50)
