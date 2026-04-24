#!/usr/bin/env python
"""
Simple PyTorch Dataset for LeRobot-style robotics data.
This is a minimal implementation showing the core concepts.
"""

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image
from torchvision.io import read_video


class SimpleRobotDataset(data.Dataset):
    """
    A simple PyTorch Dataset for robotics data, inspired by LeRobotDataset.
    
    Expected directory structure:
    root/
    ├── meta/
    │   ├── info.json          # Dataset metadata
    │   ├── stats.json         # Normalization statistics
    │   └── episodes.parquet   # Episode boundaries
    ├── videos/                # Video files
    │   └── observation.images.{camera_name}/
    │       └── chunk-000/
    │           └── file-000.mp4
    └── data/
        └── chunk-000/
            └── file-000.parquet
    """

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        image_transform: Callable | None = None,
        obs_delta_indices: list[int] | None = None,
        action_delta_indices: list[int] | None = None,
        fps: int = 30,
    ):
        """
        Args:
            root: Root directory of the dataset
            transform: Optional transform for the entire batch
            image_transform: Optional transform for images (applied after decode)
            obs_delta_indices: Observation frame indices (relative offsets)
            action_delta_indices: Action frame indices (relative offsets)
            fps: Frames per second for timestamp calculation
        """
        self.root = Path(root)
        self.transform = transform
        self.image_transform = image_transform
        self.obs_delta_indices = obs_delta_indices or [0]
        self.action_delta_indices = action_delta_indices or list(range(50))
        self.fps = fps

        self._load_metadata()
        self._load_episode_boundaries()
        self._load_data()
        self._setup_image_transforms()

    def _load_metadata(self):
        """Load dataset metadata from info.json and stats.json."""
        with open(self.root / "meta" / "info.json") as f:
            info = json.load(f)
        
        self.features = info["features"]
        self.dataset_fps = info["fps"]
        self.total_episodes = info["total_episodes"]
        
        with open(self.root / "meta" / "stats.json") as f:
            self.stats = json.load(f)

    def _load_episode_boundaries(self):
        """Load episode boundaries from parquet file."""
        episodes_path = self.root / "meta" / "episodes.parquet"
        if episodes_path.exists():
            self.episodes_df = pd.read_parquet(episodes_path)
        else:
            self.episodes_df = None

        self.num_frames = self.features.get("index", {}).get("count", [0])[0]
        if self.num_frames == 0:
            self.num_frames = self._count_frames_from_parquet()

    def _count_frames_from_parquet(self) -> int:
        """Count total frames by loading parquet files."""
        total = 0
        data_dir = self.root / "data"
        if data_dir.exists():
            for chunk_dir in sorted(data_dir.glob("chunk-*")):
                for parquet_file in sorted(chunk_dir.glob("*.parquet")):
                    df = pd.read_parquet(parquet_file)
                    total += len(df)
        return total

    def _load_data(self):
        """Load all parquet data into memory for simplicity."""
        self.data = {}
        data_dir = self.root / "data"
        
        if not data_dir.exists():
            return

        all_dfs = []
        for chunk_dir in sorted(data_dir.glob("chunk-*")):
            for parquet_file in sorted(chunk_dir.glob("*.parquet")):
                df = pd.read_parquet(parquet_file)
                all_dfs.append(df)
        
        if all_dfs:
            self.data = pd.concat(all_dfs, ignore_index=True)
            
            self.frame_index_to_row = {}
            for idx, row in enumerate(self.data.itertuples()):
                self.frame_index_to_row[int(row.index)] = idx

    def _setup_image_transforms(self):
        """Setup default image transforms."""
        if self.image_transform is None:
            self.image_transform = T.Compose([
                T.Resize((224, 224)),
            ])

    def _get_frame_timestamp(self, frame_index: int) -> float:
        """Get timestamp for a given frame index."""
        return frame_index / self.fps

    def _resolve_delta_indices(
        self, 
        current_index: int, 
        delta_indices: list[int]
    ) -> list[int]:
        """Resolve relative delta indices to absolute frame indices."""
        return [current_index + delta for delta in delta_indices]

    def _load_image(self, video_path: str | Path, frame_timestamp: float) -> torch.Tensor:
        """
        Load a single frame from video at the specified timestamp.
        
        Args:
            video_path: Path to the video file
            frame_timestamp: Timestamp in seconds for the frame to extract
            
        Returns:
            Tensor of shape (C, H, W) with dtype uint8
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            return torch.zeros(3, 480, 640, dtype=torch.uint8)
        
        video, _, _ = read_video(str(video_path), pts_unit="sec")
        frame_idx = int(frame_timestamp * self.dataset_fps)
        frame_idx = min(frame_idx, len(video) - 1)
        
        frame = video[frame_idx]
        frame = frame.permute(2, 0, 1)
        
        return frame

    def __len__(self) -> int:
        """Return the total number of frames."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single frame with all observations and actions.
        
        Returns:
            Dictionary containing:
            - observation.state: Robot state (n_obs_steps, state_dim)
            - observation.images.{camera}: Image tensors
            - action: Action vector (n_action_steps, action_dim)
            - action_is_pad: Boolean mask for padded actions
            - timestamp: Frame timestamp
            - frame_index: Frame index
            - episode_index: Episode index
        """
        row = self.data.iloc[idx]
        current_frame_index = int(row["frame_index"])
        current_episode_index = int(row["episode_index"])
        current_timestamp = float(row["timestamp"])

        obs_indices = self._resolve_delta_indices(current_frame_index, self.obs_delta_indices)
        action_indices = self._resolve_delta_indices(current_frame_index, self.action_delta_indices)

        obs_states = []
        for obs_idx in obs_indices:
            obs_row = self.data.iloc[self.frame_index_to_row.get(obs_idx, idx)]
            obs_states.append(torch.tensor(obs_row["observation.state"], dtype=torch.float32))
        observation_state = torch.stack(obs_states) if len(obs_states) > 1 else obs_states[0].unsqueeze(0)

        images = {}
        for key in self.features.keys():
            if key.startswith("observation.images."):
                camera_name = key.replace("observation.images.", "")
                
                video_rel_path = self.features[key]["info"].get("video_path", "")
                video_path = self.root / "videos" / camera_name / video_rel_path
                
                frame_timestamp = self._get_frame_timestamp(obs_indices[-1])
                img = self._load_image(video_path, frame_timestamp)
                
                if self.image_transform:
                    img = self.image_transform(img)
                
                images[key] = img

        actions = []
        action_is_pad = []
        for action_idx in action_indices:
            if action_idx in self.frame_index_to_row:
                action_row = self.data.iloc[self.frame_index_to_row[action_idx]]
                if int(action_row["episode_index"]) == current_episode_index:
                    actions.append(torch.tensor(action_row["action"], dtype=torch.float32))
                    action_is_pad.append(False)
                else:
                    actions.append(torch.zeros_like(torch.tensor(row["action"], dtype=torch.float32)))
                    action_is_pad.append(True)
            else:
                actions.append(torch.zeros_like(torch.tensor(row["action"], dtype=torch.float32)))
                action_is_pad.append(True)
        
        action = torch.stack(actions)
        action_is_pad = torch.tensor(action_is_pad, dtype=torch.bool)

        sample = {
            "observation.state": observation_state,
            "action": action,
            "action_is_pad": action_is_pad,
            "timestamp": torch.tensor([current_timestamp], dtype=torch.float32),
            "frame_index": torch.tensor([current_frame_index], dtype=torch.int64),
            "episode_index": torch.tensor([current_episode_index], dtype=torch.int64),
        }
        
        for key, img in images.items():
            sample[key] = img

        if self.transform:
            sample = self.transform(sample)

        return sample


class Normalize:
    """Normalize observations using dataset statistics."""
    
    def __init__(self, stats: dict):
        self.stats = stats
    
    def __call__(self, sample: dict) -> dict:
        for key, value in sample.items():
            if key in self.stats and isinstance(value, torch.Tensor):
                stat = self.stats[key]
                if "mean" in stat and "std" in stat:
                    mean = torch.tensor(stat["mean"], dtype=value.dtype)
                    std = torch.tensor(stat["std"], dtype=value.dtype)
                    sample[key] = (value - mean) / (std + 1e-8)
        return sample


class ToDevice:
    """Move tensors to device."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    def __call__(self, sample: dict) -> dict:
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                sample[key] = value.to(self.device)
        return sample


class Collator:
    """Collate function for batching."""
    
    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value
    
    def __call__(self, batch: list[dict]) -> dict:
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            if key == "action_is_pad":
                collated[key] = torch.stack([item[key] for item in batch])
            elif key.startswith("observation.images."):
                collated[key] = torch.stack([item[key] for item in batch])
            elif isinstance(batch[0][key], torch.Tensor):
                if batch[0][key].ndim == 0:
                    collated[key] = torch.stack([item[key].unsqueeze(0) for item in batch])
                else:
                    collated[key] = torch.stack([item[key] for item in batch])
        
        return collated


def create_simple_dataset(
    root: str,
    batch_size: int = 1,
    num_workers: int = 4,
    device: str = "cuda",
    obs_delta_indices: list[int] | None = None,
    action_delta_indices: list[int] | None = None,
) -> tuple[data.DataLoader, dict]:
    """
    Create a simple PyTorch DataLoader from a LeRobot-style dataset.
    
    Args:
        root: Path to dataset root
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        device: Device to load data to
        obs_delta_indices: Observation frame offsets
        action_delta_indices: Action frame offsets
        
    Returns:
        Tuple of (DataLoader, stats dictionary)
    """
    with open(Path(root) / "meta" / "stats.json") as f:
        stats = json.load(f)
    
    transform = None
    
    dataset = SimpleRobotDataset(
        root=root,
        transform=transform,
        obs_delta_indices=obs_delta_indices,
        action_delta_indices=action_delta_indices,
    )
    
    collate_fn = Collator()
    
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )
    
    return dataloader, stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/svla_so101_pickplace")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    
    dataloader, stats = create_simple_dataset(
        root=args.root,
        batch_size=args.batch_size,
        obs_delta_indices=[-1, 0],
        action_delta_indices=[0, 1],
        num_workers=0,
    )
    
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Stats keys: {list(stats.keys())}")
    
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        break