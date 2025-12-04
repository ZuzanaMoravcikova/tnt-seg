# TNT Segmentation Pipeline – User Manual

This repository contains a deep-learning pipeline for segmenting tunneling nanotubes (TNTs) in multidimensional fluorescence microscopy data.  
It includes training scripts, evaluation tools, and utilities for saving and analyzing predictions.

---
## Acknowledgements

This project uses and adapts parts of code from the [CS2-Net](https://github.com/iMED-Lab/CS-Net) repository,
which is licensed under the MIT License (c) 2020 ineedzx.

---

## Installation

### System requirements
- CUDA-enabled NVIDIA GPU  
- Python 3.10–3.13 with virtual environment support

### Setup

1. **Create and activate a virtual environment**
   ```bash
   python3 -m venv tnt_seg
   source tnt_seg/bin/activate
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Enable Weights & Biases logging**
   ```bash
   pip install wandb
   ```

---

## Running the training

Training is performed using `train.py`, which exposes many configurable command-line arguments.

### Basic training command
```bash
python train.py --data_path /path/to/dataset
```

This will:

- load the TNT dataset,
- initialize the 3D CSNet model,
- apply default augmentations,
- train for 200 epochs,
- save best checkpoints by validation loss and Jaccard score.

### Common arguments

- `--data_path`: dataset root directory  
- `--test_quadrant`: quadrant reserved for testing (`0`, `1`, `2`, `3`)  
- `--epochs`: number of training epochs (default: `200`)  
- `--batch_size`: batch size (default: `4`)  
- `--lr`: learning rate (default: `1e-4`)  
- `--pretrained_ckpt`: path to a pretrained checkpoint  
- `--logging`: enable logging (`1` = WandB/CSV)  
- `--ckpt_path`: directory where checkpoints are saved  

### Example – custom training run

```bash
python train.py \
    --data_path /path/to/data \
    --epochs 150 \
    --batch_size 2 \
    --lr 5e-5 \
    --logging 1 \
    --ckpt_path checkpoints_experiment1
```

### Logging and checkpoints

If logging is enabled, the script will:

- log training progress to Weights & Biases (if installed),
- otherwise fall back to CSV logging,
- automatically store the best models in:

  - `<ckpt_path>/val_loss/` – best validation loss  
  - `<ckpt_path>/val_jaccard/` – best Jaccard score  

---

## Running evaluation

After training, the model is automatically evaluated on the test set.

### Manual evaluation

```bash
python test.py \
  --data_path /path/to/data \
  --ckpt_file checkpoints_thesis/quadrant_0.ckpt \
  --test_quadrant 0
```

**Example input and output predictions of the model on *Image 1*.**  
Each row corresponds to a different cross-validation quadrant.  
To reproduce the predictions in row *i*, run:
```bash
python test.py --data_path /path/to/data --ckpt_file checkpoints_thesis/quadrant_i.ckpt --test_quadrant i
```
<img src="images/input-output_examples.png" width="600">

---

## Saving predictions

During evaluation, the script can optionally save a subset of test samples  
(input image, ground truth, and prediction) as NumPy volumes.

### Key arguments

- `--num_samples_to_save`  
  Number of test samples to save (default: `2`). Set to `0` to disable saving.

- `--save_samples_path`  
  Base directory where samples will be stored (default: `./results`).

### Output directory structure

```
<save_samples_path>/<ckpt_file name>/
    ├── image/
    ├── gt/
    └── prediction/
```

Each saved volume is stored as a `.npy` file named:

```
sample_k.npy
```

These can be loaded in Python or visualized in **napari**.

### Example: save predictions during evaluation

```bash
python test.py \
    --data_path /path/to/data \
    --ckpt_file ./checkpoints_thesis/quadrant_0.ckpt \
    --test_quadrant 0 \
    --num_samples_to_save 2 \
    --save_samples_path ./results
```

---

## Evaluating saved predictions

The repository includes a standalone evaluation tool:  
`eval_from_file.py`.

This script evaluates predictions that were saved to disk and computes all quantitative metrics used in this work (Accuracy, Sensitivity, Specificity, Precision, IoU, HD, HD95).

### Why this is useful

- Works for **any method**, not only this pipeline  
- Only requires:
  - predicted segmentations  
  - reference segmentations  
- Ensures **consistent evaluation protocol** across different approaches  


### Evaluate a directory of saved predictions

```bash
python eval_from_file.py eval \
  --path ./results/quadrant_0 \
  --print_all
```

---

## Computing metrics across multiple runs

Useful for cross-validation or comparing multiple model variants.

### Aggregation command

```bash
python eval_from_file.py aggregate \
    --root_dir ./results \
    --sub_dirs quadrant_0 quadrant_1 quadrant_2 quadrant_3
```

This will:

- evaluate each run,
- print a formatted table with metrics,
- compute overall **mean** and **standard deviation** across runs.

The aggregated results were used in the thesis to produce the cross-validation tables.

