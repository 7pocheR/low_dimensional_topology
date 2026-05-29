#!/usr/bin/env python3
"""
Step 1: Pre-compute 40x augmented bird-deer dataset with deterministic seeding.
Uses the SAME torchvision augmentation as CNN training (mild, standard).
Saves images + labels + original indices + 3D PCA coordinates in a single NPZ.
"""

import os
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image
import argparse
import time
from pathlib import Path

BIRD_CLASS = 2
DEER_CLASS = 4
BASE_SEED = 12345
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = str(PROJECT_ROOT / 'data')
DEFAULT_OUTPUT = str(PROJECT_ROOT / 'data' / 'cifar10_bird_deer_40x.npz')


def create_augmented_dataset(n_aug=40, data_dir=DEFAULT_DATA_DIR,
                              output_path=None):
    print("=" * 60)
    print(f"Creating {n_aug}x augmented bird-deer dataset")
    print("=" * 60)

    # Load CIFAR-10 training set
    train_full = datasets.CIFAR10(root=data_dir, train=True, download=True)
    test_full = datasets.CIFAR10(root=data_dir, train=False, download=True)

    # Filter bird and deer
    train_mask = [i for i in range(len(train_full))
                  if train_full.targets[i] in [BIRD_CLASS, DEER_CLASS]]
    test_mask = [i for i in range(len(test_full))
                 if test_full.targets[i] in [BIRD_CLASS, DEER_CLASS]]

    train_images = train_full.data[train_mask]  # (N_train, 32, 32, 3) uint8
    train_labels_raw = np.array(train_full.targets)[train_mask]
    train_labels = (train_labels_raw == DEER_CLASS).astype(np.int64)  # 0=bird, 1=deer

    test_images = test_full.data[test_mask]
    test_labels_raw = np.array(test_full.targets)[test_mask]
    test_labels = (test_labels_raw == DEER_CLASS).astype(np.int64)
    test_orig_indices = np.array(test_mask)

    n_train = len(train_images)
    print(f"Training images: {n_train} (bird + deer)")
    print(f"Test images: {len(test_images)}")
    print(f"Augmentation factor: {n_aug}x")
    print(f"Total augmented: {n_train * n_aug}")

    # Define augmentation (same as CNN training)
    aug_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])

    # Generate augmented images with deterministic seeding
    print("\nGenerating augmented images...")
    t0 = time.time()

    all_images = []
    all_labels = []
    all_orig_indices = []
    all_aug_indices = []

    for img_idx in range(n_train):
        if img_idx % 1000 == 0:
            print(f"  {img_idx}/{n_train} ({time.time()-t0:.1f}s)")

        img_uint8 = train_images[img_idx]  # (32, 32, 3) uint8
        label = train_labels[img_idx]
        orig_idx = train_mask[img_idx]
        pil_img = Image.fromarray(img_uint8)

        for aug_idx in range(n_aug):
            torch.manual_seed(BASE_SEED + img_idx * n_aug + aug_idx)
            aug_img = aug_transform(pil_img)
            aug_np = np.array(aug_img)  # (32, 32, 3) uint8

            all_images.append(aug_np)
            all_labels.append(label)
            all_orig_indices.append(orig_idx)
            all_aug_indices.append(aug_idx)

    all_images = np.array(all_images, dtype=np.uint8)  # (N*n_aug, 32, 32, 3)
    all_labels = np.array(all_labels, dtype=np.int64)
    all_orig_indices = np.array(all_orig_indices, dtype=np.int64)
    all_aug_indices = np.array(all_aug_indices, dtype=np.int64)

    print(f"Augmented dataset: {all_images.shape} ({time.time()-t0:.1f}s)")

    # Compute PCA
    print("\nComputing PCA...")
    flat = all_images.reshape(len(all_images), -1).astype(np.float32) / 255.0

    scaler = StandardScaler()
    flat_scaled = scaler.fit_transform(flat)

    pca = PCA(n_components=3)
    pca_3d_raw = pca.fit_transform(flat_scaled)

    # Normalize PCA output to unit variance per component (matches original pipeline)
    pca_std = pca_3d_raw.std(axis=0)
    pca_3d = (pca_3d_raw / pca_std).astype(np.float32)

    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"PCA std per component: {pca_std}")
    print(f"PCA 3D shape: {pca_3d.shape}")

    # Also compute PCA for test set (same normalization)
    test_flat = test_images.reshape(len(test_images), -1).astype(np.float32) / 255.0
    test_flat_scaled = scaler.transform(test_flat)
    test_pca_3d = (pca.transform(test_flat_scaled) / pca_std).astype(np.float32)

    # Save
    print(f"\nSaving to {output_path}...")
    np.savez_compressed(output_path,
        # Augmented training data
        images=all_images,
        labels=all_labels,
        original_indices=all_orig_indices,
        aug_indices=all_aug_indices,
        pca_3d=pca_3d,
        # PCA parameters (for reproducing projection)
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        pca_explained_variance=pca.explained_variance_ratio_,
        pca_std=pca_std,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        # Test data (unaugmented)
        test_images=test_images,
        test_labels=test_labels,
        test_orig_indices=test_orig_indices,
        test_pca_3d=test_pca_3d,
        # Metadata
        n_aug=np.array(n_aug),
        base_seed=np.array(BASE_SEED),
        bird_class=np.array(BIRD_CLASS),
        deer_class=np.array(DEER_CLASS),
    )

    fsize = os.path.getsize(output_path) / 1e9
    print(f"Saved ({fsize:.2f} GB)")
    print(f"Total time: {time.time()-t0:.1f}s")

    # Print summary stats
    n_bird = (all_labels == 0).sum()
    n_deer = (all_labels == 1).sum()
    print(f"\nSummary: {n_bird} bird + {n_deer} deer = {len(all_labels)} total")
    print(f"PCA range: {pca_3d.min(axis=0)} to {pca_3d.max(axis=0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-aug', type=int, default=40)
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output', type=str,
                        default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    create_augmented_dataset(args.n_aug, args.data_dir, args.output)
