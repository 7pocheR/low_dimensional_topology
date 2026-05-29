#!/usr/bin/env python3
"""
Visualize detected witness cycles in 3D PCA space.
Produces a figure similar to Figure 7 in the paper.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path


def visualize(dataset_path, output_dir):
    data = np.load(dataset_path, allow_pickle=True)
    pca_3d = data['pca_3d']
    labels = data['labels']

    link_path = os.path.join(output_dir, 'link_detection_result.json')
    with open(link_path) as f:
        link = json.load(f)

    # Support multi-link and single-link format
    if 'links' in link:
        links = link['links']
        all_bird = np.array(link['all_bird_global_indices'])
        all_deer = np.array(link['all_deer_global_indices'])
    else:
        # Single-link format: wrap as a one-element list
        links = [{
            'bird_global': link['bird_cycle_global_indices'],
            'deer_global': link['deer_cycle_global_indices'],
        }]
        all_bird = np.array(link['bird_cycle_global_indices'])
        all_deer = np.array(link['deer_cycle_global_indices'])

    bird_mask = labels == 0
    deer_mask = labels == 1

    # --- Plot 1: Full point cloud with witness cycles highlighted ---
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Subsample background for visibility
    n_bg = min(5000, bird_mask.sum(), deer_mask.sum())
    np.random.seed(42)
    bird_bg = np.random.choice(np.where(bird_mask)[0], n_bg, replace=False)
    deer_bg = np.random.choice(np.where(deer_mask)[0], n_bg, replace=False)

    ax.scatter(pca_3d[bird_bg, 0], pca_3d[bird_bg, 1], pca_3d[bird_bg, 2],
               c='lightblue', s=1, alpha=0.1, label='bird (bg)')
    ax.scatter(pca_3d[deer_bg, 0], pca_3d[deer_bg, 1], pca_3d[deer_bg, 2],
               c='lightsalmon', s=1, alpha=0.1, label='deer (bg)')

    # Plot witness cycles
    colors_bird = plt.cm.Blues(np.linspace(0.5, 1.0, max(len(links), 1)))
    colors_deer = plt.cm.Reds(np.linspace(0.5, 1.0, max(len(links), 1)))

    for li, lnk in enumerate(links[:10]):  # show up to 10 links
        b_idx = np.array(lnk['bird_global'])
        d_idx = np.array(lnk['deer_global'])
        b_pts = pca_3d[b_idx]
        d_pts = pca_3d[d_idx]

        # Plot cycle as connected polyline
        ax.plot(b_pts[:, 0], b_pts[:, 1], b_pts[:, 2],
                color=colors_bird[li], linewidth=1.5, alpha=0.8)
        ax.plot(d_pts[:, 0], d_pts[:, 1], d_pts[:, 2],
                color=colors_deer[li], linewidth=1.5, alpha=0.8)

        # Highlight vertices
        ax.scatter(b_pts[:, 0], b_pts[:, 1], b_pts[:, 2],
                   c=[colors_bird[li]], s=15, zorder=5)
        ax.scatter(d_pts[:, 0], d_pts[:, 1], d_pts[:, 2],
                   c=[colors_deer[li]], s=15, zorder=5)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    n_links = len(links)
    ax.set_title(f'Witness Cycles in 3D PCA Space ({n_links} linked pairs)')

    save_path = os.path.join(output_dir, 'witness_cycles_3d.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()

    # --- Plot 2: Zoomed into witness region ---
    if len(all_bird) > 0 and len(all_deer) > 0:
        all_witness = np.concatenate([all_bird, all_deer])
        witness_pts = pca_3d[all_witness]
        center = witness_pts.mean(axis=0)
        radius = np.linalg.norm(witness_pts - center, axis=1).max() * 1.5

        fig2 = plt.figure(figsize=(14, 10))
        ax2 = fig2.add_subplot(111, projection='3d')

        # Plot nearby background points
        dists = np.linalg.norm(pca_3d - center, axis=1)
        nearby = dists < radius * 2
        nearby_bird = nearby & bird_mask
        nearby_deer = nearby & deer_mask

        ax2.scatter(pca_3d[nearby_bird, 0], pca_3d[nearby_bird, 1], pca_3d[nearby_bird, 2],
                    c='lightblue', s=3, alpha=0.2, label=f'bird ({nearby_bird.sum()})')
        ax2.scatter(pca_3d[nearby_deer, 0], pca_3d[nearby_deer, 1], pca_3d[nearby_deer, 2],
                    c='lightsalmon', s=3, alpha=0.2, label=f'deer ({nearby_deer.sum()})')

        for li, lnk in enumerate(links[:10]):
            b_pts = pca_3d[np.array(lnk['bird_global'])]
            d_pts = pca_3d[np.array(lnk['deer_global'])]
            ax2.plot(b_pts[:, 0], b_pts[:, 1], b_pts[:, 2],
                     color='blue', linewidth=2, alpha=0.9)
            ax2.plot(d_pts[:, 0], d_pts[:, 1], d_pts[:, 2],
                     color='red', linewidth=2, alpha=0.9)
            ax2.scatter(b_pts[:, 0], b_pts[:, 1], b_pts[:, 2], c='blue', s=20, zorder=5)
            ax2.scatter(d_pts[:, 0], d_pts[:, 1], d_pts[:, 2], c='red', s=20, zorder=5)

        ax2.set_xlim(center[0] - radius, center[0] + radius)
        ax2.set_ylim(center[1] - radius, center[1] + radius)
        ax2.set_zlim(center[2] - radius, center[2] + radius)
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_zlabel('PC3')
        ax2.set_title('Zoomed: Witness Cycles Region')
        ax2.legend()

        save_path2 = os.path.join(output_dir, 'witness_cycles_3d_zoomed.png')
        plt.savefig(save_path2, dpi=200, bbox_inches='tight')
        print(f"Saved to {save_path2}")
        plt.close()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=str(project_root / 'data' / 'cifar10_bird_deer_40x.npz'))
    parser.add_argument('--output-dir', default=os.environ.get('WITNESS_OUTPUT_DIR',
                        str(project_root / 'results' / 'witness_eval')))
    args = parser.parse_args()
    visualize(args.dataset, args.output_dir)
