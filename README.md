# Low-Dimensional Topology of Deep Neural Networks: Experiment Code

Code release for:

**Low-dimensional topology of deep neural networks**

Junyu Ren and Lek-Heng Lim

ICML 2026 camera-ready version

- Paper: forthcoming
- Code archive: https://github.com/7pocheR/low_dimensional_topology
- Contact: Junyu Ren `<junyuren@uchicago.edu>`, Lek-Heng Lim `<lekheng@uchicago.edu>`
- License: MIT, see `LICENSE`.

This folder is a clean staging area for the code release accompanying the ICML camera-ready paper. It is organized by paper experiment: each `exp_*` directory corresponds to one paper section, table, or figure family.

The folder intentionally excludes generated logs, model checkpoints, large datasets, raw result dumps, manuscript files, and historical exploratory scripts. Camera-ready authorship anonymity is not required, but the folder is kept focused on reproducibility.

## Layout

| Folder | Paper location | Target artifact | Purpose |
| --- | --- | --- | --- |
| `exp_1_table2_relu_gelu_hopf` | Section 6.2, Table 2 | ReLU vs GELU Hopf-link accuracies | Width-3 feedforward networks on the thickened Hopf link. |
| `exp_2_table3_plain_relu_vs_resnet_hopf` | Section 6.3, Table 3 | Plain ReLU vs ResNet Hopf-link accuracies | Tests skip connections as an escape mechanism. |
| `exp_3_resnet_disk_mechanism` | Section 6.4 and Appendix ResNet visualization | ResNet disk-annulus mechanism figure | Produces the layer/block visualization of the learned folding map. |
| `exp_4_higher_dim_r5_multicopy` | Section 6.5, Table 4 | Multi-copy `S^2 sqcup S^2` in `R^5` | Tests higher-dimensional linking with increasing total linking number. |
| `exp_5_width_expansion_r7` | Appendix width expansion | Width-expansion table in `R^7` | Shows that increasing width removes the bottleneck obstruction. |
| `exp_6_layer_tracking` | Appendix layer tracking | Layer-by-layer linking/min-distance table | Tracks linking number and minimum inter-class distance through trained networks. |
| `exp_7_cifar10_link_detection` | Section 6.6 and Appendix CIFAR/link-detection details | CIFAR-10 PCA-3D link witnesses and consistency summaries | Detects linked graph cycles in PCA-projected CIFAR-10 augmentations. |
| `exp_8_cifar10_classification` | Section 6.6 and Appendix CIFAR/confusion details | Binary classification table, local link-gap table, 10-class confusion correlations | Trains width-bounded CNNs and compares linking consistency to classification difficulty. |
| `shared` | Tables 2-4 | Aggregation helper | Converts per-run JSON files into paper table summaries. |

## Experiment Details

### `exp_1_table2_relu_gelu_hopf`

Paper claim: width-3 ReLU networks remain below the topological ceiling on a thickened Hopf link, while GELU can break the monotonicity obstruction.

Files:

- `run_table2_trial.py`: one `(depth, activation, seed)` trial.
- `submit_table2.sh`: SLURM array launcher.

Paper configuration:

- Activations: `relu`, `gelu`.
- Full run depths: `3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20`.
- Displayed table depths: `3, 5, 8, 12, 16, 20`.
- Seeds: 30 per cell.
- Optimizer: Adam, learning rate `1e-3`, early stopping.

### `exp_2_table3_plain_relu_vs_resnet_hopf`

Paper claim: ReLU ResNets overcome the Hopf-link obstruction even at width 3, while plain ReLU networks remain limited.

Files:

- `run_table3_trial.py`: one `(depth, architecture, seed)` trial.
- `submit_table3.sh`: SLURM array launcher for the paper run.

Paper configuration:

- Architectures: `relu` plain feedforward, `resnet`.
- Depths: `3, 4, 5, 6, 7, 8`.
- Seeds: 10 per cell in the current paper.

### `exp_3_resnet_disk_mechanism`

Paper claim: a width-2 ReLU ResNet learns a folding mechanism equivalent to the identity `|x| = x + 2 ReLU(-x)`.

Files:

- `colorful_resnet_disk.py`: trains and visualizes the disk-annulus / point-circle mechanism.
- `run_resnet_disk.sh`: SLURM launcher.

Paper configuration:

- Width-2 ReLU ResNet.
- Three residual blocks.
- Disk-annulus separation task in `R^2`.
- Seed 103 is the displayed run.

### `exp_4_higher_dim_r5_multicopy`

Paper claim: in `R^5`, multiple disjoint copies of linked `S^2 sqcup S^2` create multiple local entanglement regions; non-monotonic activations become more helpful as the copy count grows.

Files:

- `train_width_scaling_v7.py`: training script for linked-sphere experiments.
- `submit_linking_v7.sh`: SLURM array launcher.
- `analyze_linking_scaling.py`, `plot_linking_results.py`: analysis and plotting utilities.
- `generate_linked_spheres_dataset.py`: linked-sphere dataset helper.

Paper configuration:

- Ambient space: `R^5`.
- Manifolds: `S^2 sqcup S^2`.
- Width 5, depth 5.
- Copies: `k = 1, 2, 5, 10, 20, 50`.
- Models: ReLU, ReLU+Skip, GELU, Swish.
- Seeds: 100 per condition.

### `exp_5_width_expansion_r7`

Paper claim: expanding width past the critical bottleneck threshold removes the obstruction.

Files:

- `train_width_scaling_v7.py`: training script.
- `submit_width_expansion_r7.sh`: SLURM array launcher.
- `aggregate_width_scaling_results.py`: summary utility.

Paper configuration:

- Ambient space: `R^7`.
- Manifolds: `S^3 sqcup S^3`.
- Copies: `k = 10`.
- Depth: 5.
- Widths: `7, 8, 10, 14, 21, 28, 35, 49`.
- Seeds: 15 per width.

### `exp_6_layer_tracking`

Paper claim: plain ReLU destroys disjointness while GELU and ReLU+skip unlink while preserving separation.

Files:

- `track_linking_through_layers.py`: computes layer-by-layer linking and minimum distance.
- `submit_linking_tracking.sh`: SLURM launcher.

Paper configuration:

- Hopf-link task.
- Width 3, depth 5.
- Models: ReLU, GELU, ReLU+skip.
- Evaluation: best seed, 200 points per class.

### `exp_7_cifar10_link_detection`

Paper claim: CIFAR-10 augmented class clouds projected to PCA-3D contain reproducible graph-cycle linking signals.

Files:

- `src/link_detector.py`: kNN graph, cycle-basis, and Gauss-linking routines.
- `utils/`: CIFAR-10 loading, augmentation, PCA, and graph helpers.
- `scripts/prepare_augmented_dataset.py`: builds augmented PCA-3D data.
- `scripts/detect_link_augmented.py`: detects linked cycles.
- `scripts/visualize_witness_3d.py`: visualizes detected witness cycles.
- `analysis/eps_min_all_pairs_augmented.py`: all-pair threshold search / consistency support.
- `analysis/eps_min_refine.py`: optional refinement of the augmented all-pair thresholds.
- `analysis/aggregate_eps_min.py`: merges sharded augmented all-pair outputs.
- `analysis/analyze_epsilon_scale.py`: reports the detected epsilon scale relative to the PCA-3D point cloud.
- `analysis/validate_witness_cycles.py`: optional witness-cycle interpolation/image validation.
- `analysis/visualize_linked_cycles.py`: auxiliary full-cycle visualization for detected links.
- `scripts/slurm_eps_min_augmented.sh`: single-job augmented all-pair run.
- `scripts/slurm_eps_min_array.sh`: sharded augmented all-pair run.
- `scripts/slurm_validate_witness.sh`: launcher for witness-cycle validation.

Paper configuration:

- Dataset: CIFAR-10 train set with 20x augmentation.
- Size: 1.05M augmented samples total, 105k per class.
- Projection: flatten, standardize, PCA to `R^3`.
- Graph: mutual kNN, `k=15`, minimum cycle length 30.
- Displayed witness: bird-deer, `epsilon approx 0.034`, linking number `-1`.
- Consistency: 11 runs over all 45 class pairs with fixed epsilon sequence.

### `exp_8_cifar10_classification`

Paper claim: CIFAR-10 linking consistency correlates with pairwise classification difficulty, and the non-monotonic advantage localizes near a detected link witness.

Files:

- `scripts/train_cnn_binary.py`: binary linked/unlinked classification.
- `scripts/train_cnn_10class.py`: 10-class CNN runs.
- `scripts/train_cnn_unlinked.py`: unlinked-control training script.
- `scripts/retrain_and_eval_witness.py`, `scripts/eval_witness_subsets.py`: local link-gap diagnostic.
- `analysis/analyze_linking_confusion_correlation.py`: 10-class confusion vs linking consistency.
- `scripts/analyze_distance_vs_linking.py`, `scripts/distance_analysis_correct.py`: distance-metric controls.
- `utils/`: CIFAR-10 loading and augmentation helpers.
- `scripts/slurm_*.sh`: cluster launchers for classification and witness diagnostics.

Paper configuration:

- CNNs are width-bounded so every flattened intermediate representation has dimension at most 3072.
- Activations: ReLU, ELU, SELU, LeakyReLU, GELU, Swish, Mish.
- Depth variants: L5, L8, L11.
- Skip/no-skip variants for 10-class runs.
- Binary linked pair: deer-dog.
- Binary unlinked control: frog-ship.
- Local-gap diagnostic: bird-deer L8 no-skip ReLU/GELU, stratified by distance to the detected PCA-3D witness.

## Path Conventions

Scripts default to local `data/` and `results/` subfolders inside their own experiment directory. SLURM launchers do not request a specific GPU model; they only request the number of GPUs used by the original jobs. On clusters, override paths with environment variables such as `DATA_DIR`, `OUTPUT_DIR`, `RESULTS_DIR`, `RESULTS_BASE`, `CIFAR10_DATA_DIR`, and `CONDA_ENV`.

## Environment

Install the core Python dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

The synthetic experiments use `numpy`, `scipy`, `scikit-learn`, `matplotlib`, and `torch`. The CIFAR-10 experiments additionally use `torchvision`, `pandas`, `seaborn`, and `pillow`; `tensorflow` is only an optional fallback for CIFAR-10 downloading if `torchvision` is unavailable.

## Smoke Tests

These commands run tiny CPU-compatible jobs and only check that the scripts execute end-to-end; they do not reproduce paper numbers.

```bash
python3 exp_1_table2_relu_gelu_hopf/run_table2_trial.py \
  --depth 3 --activation relu --seed 0 \
  --epochs 2 --patience 2 --n_points_per_curve 100 --batch_size 32 \
  --output_dir /tmp/topo_camera_smoke/table2

python3 exp_2_table3_plain_relu_vs_resnet_hopf/run_table3_trial.py \
  --depth 3 --arch resnet --seed 0 \
  --epochs 2 --patience 2 --n_points_per_curve 100 --n_train 120 --batch_size 32 \
  --output_dir /tmp/topo_camera_smoke/table3

python3 exp_4_higher_dim_r5_multicopy/train_width_scaling_v7.py \
  --n 2 --depth 2 --activation relu --num_copies 1 \
  --train_samples 256 --val_test_samples 128 \
  --epochs 2 --patience 2 --batch_size 64 \
  --output_dir /tmp/topo_camera_smoke/r5
```

## Aggregation

After full runs have produced result JSON files in the default `results/` folders, Tables 2--4 can be summarized with:

```bash
python3 shared/aggregate_tables.py
```

The width-expansion appendix experiment has its own aggregator:

```bash
python3 exp_5_width_expansion_r7/aggregate_width_scaling_results.py
```

Other CIFAR-10 and layer-tracking analyses are generated by the analysis scripts in their respective experiment folders; they are not yet wrapped by one top-level table-regeneration command.

## Duplication Policy

Experiment folders are intentionally self-contained. In particular, `exp_4_higher_dim_r5_multicopy/train_width_scaling_v7.py` and `exp_5_width_expansion_r7/train_width_scaling_v7.py` are identical by design, and the CIFAR-10 loader/augmentation utilities are duplicated between `exp_7_cifar10_link_detection` and `exp_8_cifar10_classification`. This avoids hidden cross-experiment imports when a reader runs one folder independently.

## Source Notes

Synthetic experiments were copied from the local development trees used for the camera-ready runs.

CIFAR-10 experiments were imported from the cluster working copy whose tracked source matches GitHub repo `7pocheR/link_in_machine_learning` at commit `a75842b` (`Add distinct link detection + per-link eval script`). Generated logs, cache files, results directories, and large data were not copied.

## Remaining Release Polish

- Add the public paper URL once it is available.
- Add one command that regenerates every paper and appendix table from result JSON files.

## Citation

```bibtex
@inproceedings{renlim2026lowdimensional,
  title     = {Low-dimensional topology of deep neural networks},
  author    = {Ren, Junyu and Lim, Lek-Heng},
  booktitle = {International Conference on Machine Learning},
  year      = {2026}
}
```
