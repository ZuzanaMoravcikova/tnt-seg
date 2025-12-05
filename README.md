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
## Project structure
The source code of the TNT segmentation framework is organized as follows:
```text
tnt-seg/
  ├─ checkpoints_thesis/
  │    ├─ quadrant_0.ckpt
  │    ├─ quadrant_1.ckpt
  │    ├─ quadrant_2.ckpt
  │    └─ quadrant_3.ckpt
  │
  ├─ data_preparation/
  │    ├─ __init__.py
  │    ├─ synth_tubular_dataset.py
  │    ├─ tnt_crops.py
  │    ├─ tnt_dataset_final.py
  │    ├─ tnt_io.py
  │    └─ tnt_regions.py
  │
  ├─ model/
  │    ├─ __init__.py
  │    ├─ csnet_3d.py
  │    └─ csnet_3d_lightning.py
  │
  ├─ utils/
  │    ├─ __init__.py
  │    ├─ losses.py
  │    ├─ my_metrics.py
  │    ├─ my_typing.py
  │    └─ transforms.py
  │
  ├─ .gitignore
  ├─ eval_from_file.py
  ├─ README.md
  ├─ requirements.txt
  ├─ test.py
  └─ train.py
```
The directory `checkpoints_thesis` stores pretrained model checkpoints for quadrants 0--3. 
These files can be used for running the evaluation and for reproducing the results presented in this thesis.

The directory `data_preparation` contains all code related to preparing the image data for training and testing. 
The scripts in this folder load the TNT data, create train--test splits, generate patches for training, 
and wrap this functionality into a PyTorch Lightning datamodule.

The directory `model` implements the segmentation network and its PyTorch Lightning module, 
which encapsulates the training, validation, and inference logic.

The directory `utils` collects various helper functions and definitions, including loss functions, 
evaluation metrics, type definitions, and data transformations and augmentations.

The script `train.py` is the main training entry point, while `test.py` is used to evaluate a trained model.

The script `eval_from_file.py` provides a standalone evaluation tool that operates on previously saved predictions.

---
## Data format 

The scripts expect a TNT dataset organized in a folder structure compatible with:

```text
<data_path>/
  ├─ 01/
  │    ├─ t000.tif
  │    ├─ t001.tif
  │    ├─ ...
  │    └─ t019.tif
  │
  └─ 01_GT/
        └─SEG/
            ├─ mask000.tif
            └─ mask017.tif
```

---
## Running the training

Training is performed using `train.py`, which exposes many configurable command-line arguments.

### Basic training command
```bash
python train.py --data_path </path/to/dataset> --test_quadrant <i>
```

This will:

- load the TNT dataset,
- reserve quadrant `i` for testing,
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
    --data_path </path/to/data> \
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
  --data_path </path/to/data> \
  --ckpt_file checkpoints_thesis/quadrant_0.ckpt \
  --test_quadrant 0
```

**Example input and output predictions of the model on *Image 1*.**  
Each row corresponds to a different cross-validation quadrant.  
To reproduce the predictions in row *i*, run:
```bash
python test.py --data_path </path/to/data> --ckpt_file checkpoints_thesis/quadrant_<i>.ckpt --test_quadrant <i>
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
sample_<k>.npy
```

These can be loaded in Python or visualized in **napari**.

### Example: save predictions during evaluation

```bash
python test.py \
    --data_path </path/to/data> \
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

