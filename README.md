# Learning without Forgetting (LwF) Reproduction Project

This repository contains a reproduction project for **Learning without Forgetting (LwF)** by Li and Hoiem.  
The project studies **catastrophic forgetting** in deep neural networks: when a model is trained on a new task, its performance on a previously learned task can degrade.

The reproduced continual-learning setting is:

- **Old task:** ImageNet classification
- **New tasks:**
  - CUB-200-2011 bird classification
  - MIT Indoor 67 Scenes classification
- **Backbone:** AlexNet pretrained on ImageNet
- **Compared methods:**
  - Fine-Tuning
  - Feature Extraction
  - Learning without Forgetting (LwF)
- **Additional optimization:** LwF with a cosine-normalized classifier for the MIT Indoor Scenes experiment

The original LwF idea is to learn a new task using only the new-task data, while preserving the behavior of the original model on the old task through knowledge distillation.

---

## Reproduction Scope Agreed With the Instructor

This project was designed as a **focused reproduction**, not as a full replication of the entire LwF paper.  
According to the scope agreed with the course instructor, the goal was to reproduce the main qualitative trend shown in **Table 1** of the original paper, using only two ImageNet-based experiments from that table.

The reproduced experiments are:

1. **ImageNet → CUB-200-2011**  
   The old task is ImageNet classification, and the new task is fine-grained bird classification using the CUB-200-2011 dataset.

2. **ImageNet → MIT Indoor 67 Scenes**  
   The old task is ImageNet classification, and the new task is indoor scene classification using the MIT Indoor 67 Scenes dataset.

The original paper includes many additional task pairs, datasets, architectures, and baselines. These were **outside the agreed project scope**. In particular, this project was not required to reproduce the full paper, all of Table 1, or every method reported by Li and Hoiem. The purpose was to demonstrate the same main behavior:

- **Fine-Tuning** tends to cause the strongest forgetting on the old ImageNet task.
- **Feature Extraction** preserves the old task well, but adapts less strongly to the new task.
- **Learning without Forgetting (LwF)** provides a better stability-plasticity trade-off by learning the new task while preserving more old-task knowledge than Fine-Tuning.

To keep the reproduction consistent and computationally feasible, and also according to the agreement with the course instructor, the experiments were executed using:

- a single fixed random seed: `42`
- ImageNet old-task evaluation on the first **80 validation batches**
- ImageNet batch size of `64`, meaning:

```text
80 × 64 = 5120 ImageNet validation images
```

Therefore, the reported ImageNet results should be interpreted as results on a fixed ImageNet validation subset, not as full ImageNet validation-set benchmark results.  
The same seed and the same fixed ImageNet subset were used consistently across all compared methods, which keeps the comparisons within this project fair and reproducible.

---

## Project Structure

```text
LWF reproduction project/
│
├── AI documentation/
│   ├── AI Assisted Development Documentation.pdf
│   └── AI documentation - optimization.pdf
│
├── Notebooks/
│   ├── CUB code.ipynb
│   └── scenes code.ipynb
│
├── Optimization/
│   ├── Optimization code.ipynb
│   ├── Optimization report.pdf
│   └── Table.pdf
│
├── Paper/
│   └── LWF.pdf
│
├── Report/
│   └── Report.pdf
│
└── Results/
    ├── Imagenet forgetting table.pdf
    ├── results comparison.pdf
    └── Results table.pdf
```

---

## Repository Contents

### `Notebooks/`
Contains the main executable reproduction notebooks.

| Notebook | Description |
|---|---|
| `CUB code.ipynb` | Reproduces the ImageNet → CUB-200-2011 experiment. |
| `scenes code.ipynb` | Reproduces the ImageNet → MIT Indoor 67 Scenes experiment. |

Both notebooks follow the same experimental logic:

1. Load an ImageNet-pretrained AlexNet model.
2. Add a new task-specific classification head.
3. Measure the old-task ImageNet baseline.
4. Train and evaluate Fine-Tuning.
5. Train and evaluate Feature Extraction.
6. Train and evaluate Learning without Forgetting.
7. Save result tables for report usage.

### `Optimization/`
Contains the bonus optimization experiment for ImageNet → MIT Indoor Scenes.

| File | Description |
|---|---|
| `Optimization code.ipynb` | Adds and evaluates the optimized LwF version. |
| `Optimization report.pdf` | Explains the optimization motivation, method, and results. |
| `Table.pdf` | Summary table for the optimization results. |

The optimized method replaces the standard new-task linear classifier with a **scaled cosine-normalized classifier** and applies a higher learning rate only to the new Scenes head.

### `Report/`
Contains the final project report:

```text
Report/Report.pdf
```

The report explains the motivation, theoretical background, implementation, experimental setup, results, limitations, and conclusions.

### `Results/`
Contains exported result tables and comparison figures used in the report.

### `AI documentation/`
Contains documentation of the AI-assisted development process, including debugging, implementation decisions, result interpretation, and the optimization phase.

### `Paper/`
Contains the original paper PDF used as the main scientific reference.

---

## Methods Implemented

### 1. Fine-Tuning
The pretrained AlexNet model is adapted to the new task by updating the shared representation and the new classification head.  
This usually improves adaptation to the new task, but it can damage old-task ImageNet performance because the shared representation changes.

### 2. Feature Extraction
The pretrained ImageNet representation is frozen, and only a new classifier is trained for the new task.  
This preserves the old task almost perfectly, but it limits adaptation to the new task.

### 3. Learning without Forgetting (LwF)
LwF uses a frozen teacher model to preserve the old task.  
The student model is trained using two losses:

```text
Total Loss = New Task Classification Loss + λ · Old Task Distillation Loss
```

In this project:

- `T = 2.0` is used as the distillation temperature.
- `λ = 1.0` is used as the old-task loss weight.
- The old task is ImageNet.
- The new task is either CUB or MIT Indoor Scenes.

The implemented LwF model uses a dual-head AlexNet design:

```text
shared AlexNet feature extractor
├── old_head: ImageNet classifier, 1000 classes
└── new_head: CUB or Scenes classifier
```

---

## Experimental Protocol

The project uses a consistent protocol across the main experiments.

| Setting | Value |
|---|---:|
| Backbone | AlexNet pretrained on ImageNet |
| Main seed | 42 |
| Warm-up stage | 1 epoch |
| Main training budget | 5 epochs |
| Optimizer | SGD |
| Base learning rate | 0.001 |
| Momentum | 0.9 |
| Distillation temperature | 2.0 |
| LwF old-task loss weight | 1.0 |
| ImageNet evaluation | First 80 validation batches |
| ImageNet batch size | 64 |
| CUB / Scenes batch size | 32 |

The ImageNet validation evaluation is limited to the first **80 batches** due to compute limitations.  
The same subset is used consistently across methods, so the comparison remains fair.

---

## Datasets

The datasets are **not included** in this repository because of size and licensing constraints.

Expected datasets:

| Dataset | Role | Notes |
|---|---|---|
| ImageNet validation set | Old-task evaluation | Used to measure forgetting. |
| CUB-200-2011 | New task | Fine-grained bird classification, 200 classes. |
| MIT Indoor 67 Scenes | New task | Indoor scene classification, 67 classes. |

The notebooks are written for Google Colab and expect the datasets to be available in Google Drive.

Default paths used in the notebooks:

```python
# ImageNet validation
/content/drive/MyDrive/lwf/imagenet_validation/imagenet_validation

# CUB-200-2011
/content/drive/MyDrive/LWF_Project/extracted/LWF_Project/CUB_200_2011

# MIT Indoor Scenes ZIP
/content/drive/MyDrive/lwf/MIT INDOOR.zip
```

If your Drive structure is different, update only the path variables in the first cells of each notebook.

---

## How to Run

### 1. Open the notebook in Google Colab
Use one of the following notebooks:

```text
Notebooks/CUB code.ipynb
Notebooks/scenes code.ipynb
Optimization/Optimization code.ipynb
```

### 2. Enable GPU
In Colab:

```text
Runtime → Change runtime type → GPU
```

### 3. Mount Google Drive
The notebooks mount Drive automatically using:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Verify dataset paths
Before training, confirm that the printed dataset paths exist.

### 5. Run the notebook from top to bottom
The notebooks are organized into clear steps:

- environment setup
- dataset preparation
- model definition
- training helpers
- baseline evaluation
- method comparison
- result export

---

## Main Reproduction Results

### ImageNet → CUB

| Method | ImageNet Old Task Accuracy | CUB New Task Accuracy |
|---|---:|---:|
| Fine-Tuning | 53.42% | 45.53% |
| Feature Extraction | 62.89% | 42.03% |
| LwF | 59.49% | 45.41% |

### ImageNet → MIT Indoor Scenes

| Method | ImageNet Old Task Accuracy | Scenes New Task Accuracy |
|---|---:|---:|
| Fine-Tuning | 48.69% | 50.82% |
| Feature Extraction | 62.89% | 55.00% |
| LwF | 59.82% | 56.27% |

### Interpretation

The reproduced results follow the expected qualitative trend:

- **Fine-Tuning** adapts to the new task but causes the strongest old-task forgetting.
- **Feature Extraction** preserves ImageNet performance because the pretrained representation is frozen, but it adapts less strongly to the new task.
- **LwF** gives a better balance between learning the new task and preserving old-task knowledge.

---

## ImageNet Forgetting Summary

The ImageNet baseline measured on the fixed 80-batch subset is:

```text
Baseline ImageNet Accuracy = 62.89%
```

| Experiment | Method | ImageNet After New Task | Δ ImageNet |
|---|---|---:|---:|
| ImageNet → CUB | Fine-Tuning | 53.42% | -9.47% |
| ImageNet → CUB | Feature Extraction | 62.89% | 0.00% |
| ImageNet → CUB | LwF | 59.49% | -3.40% |
| ImageNet → Scenes | Fine-Tuning | 48.69% | -14.20% |
| ImageNet → Scenes | Feature Extraction | 62.89% | 0.00% |
| ImageNet → Scenes | LwF | 59.82% | -3.07% |

This shows that LwF reduces catastrophic forgetting compared with Fine-Tuning, while still allowing the shared representation to adapt more than Feature Extraction.

---

## Bonus Optimization Result

The optimization experiment was performed on the ImageNet → MIT Indoor Scenes setting.

The optimized method uses:

- a scaled cosine-normalized new-task classifier
- initial scale `η0 = 30`
- a learnable scale parameter
- new-head learning rate multiplied by 5
- unchanged LwF distillation mechanism
- unchanged dataset and evaluation protocol

| Method | ImageNet Old Task Accuracy | Scenes New Task Accuracy |
|---|---:|---:|
| Fine-Tuning | 48.69% | 50.82% |
| Feature Extraction | 62.89% | 55.00% |
| Standard LwF | 59.82% | 56.27% |
| Optimized LwF | 60.72% | 58.06% |

Compared with standard LwF, the optimized method improved:

```text
Scenes accuracy:  +1.79%
ImageNet accuracy: +0.90%
```

This improvement is meaningful because the new-task accuracy increased without sacrificing old-task preservation.

---

## Requirements

The project was developed and tested in Google Colab.

Main Python libraries:

```text
torch
torchvision
numpy
pandas
Pillow
scikit-learn
matplotlib
```

A GPU runtime is strongly recommended.

---

## Notes and Limitations

- The reproduction scope was agreed with the course instructor.
- The project reproduces only the relevant part of **Table 1** from the original paper: **ImageNet → CUB-200-2011** and **ImageNet → MIT Indoor 67 Scenes**.
- The project was not intended to reproduce the entire LwF paper, all task pairs, all architectures, or all baselines.
- The goal was to show the same qualitative trend reported in the paper: Fine-Tuning forgets more, Feature Extraction preserves more but adapts less, and LwF gives a stronger balance between the old and new tasks.
- Each experiment was executed with one fixed seed (`42`), as agreed with the course instructor.
- ImageNet evaluation is performed on a fixed 80-batch validation subset, also according to the agreed project protocol.
- Results may vary slightly due to GPU availability, random initialization, and runtime environment.
- Model checkpoints are not saved by default because AlexNet checkpoints are large. To save models, set:

```python
SAVE_MODELS = True
```

---

## Authors

- Nir Moyal
- Dan Feldman

Sami Shamoon College of Engineering (SCE)

---

## Reference

Li, Z. and Hoiem, D. **Learning without Forgetting**.  
European Conference on Computer Vision (ECCV), 2016.

The original paper proposes training on the new task while preserving previous capabilities without requiring the previous task training data.
