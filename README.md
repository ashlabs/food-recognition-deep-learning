# Food Recognition with Convolutional Neural Networks

Food-101 image classification using PyTorch CNNs trained from scratch. Controlled experiments investigate channel width, label smoothing, dropout, and maximum training epochs.

The best observed configuration achieved **65.63% test accuracy**, compared with **57.39%** for the baseline.

## Results

All checkpoints were selected by minimum validation cross-entropy.

| Configuration | Selected epoch | Validation accuracy | Test accuracy | Test cross-entropy |
|---|---:|---:|---:|---:|
| Baseline CNN | 30 | 52.62% | 57.39% | 1.6429 |
| Baseline + label smoothing 0.1 | 26 | 52.83% | 58.21% | 1.6891 |
| Wider CNN | 30 | 58.01% | 63.53% | 1.4201 |
| Wider CNN + dropout 0.3, max 30 epochs | 29 | 58.58% | 63.70% | 1.3887 |
| Wider CNN + dropout 0.3, max 60 epochs | 39 | 60.20% | 65.63% | 1.3240 |

Increasing channel width produced the largest isolated improvement:
**6.14 percentage points** of test accuracy, with approximately
**3.87 times** as many trainable parameters.

The extended dropout run stopped at epoch 44 and restored epoch 39. Results reflect one seed, not averages across repeated runs.

## Controlled Comparisons

Each comparison changes one aspect relative to its reference:

| Reference | Modified configuration | Change |
|---|---|---|
| Baseline | Wider CNN | Channel widths doubled |
| Baseline | Baseline + label smoothing | Smoothing strength 0.1 |
| Wider CNN | Wider CNN + dropout | Dropout probability 0.3 |
| Wider CNN + dropout, max 30 epochs | Same model, max 60 epochs | Maximum epochs |

The baseline uses four convolutional blocks with channel widths [32, 64, 128, 256]. The wider model uses [64, 128, 256, 512]. Each block contains convolution, batch normalization, ReLU, and max pooling, followed by global average pooling and a linear classifier.

## Dataset

[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) contains 101,000 photographs across 101 food categories.

| Partition | Images | Images per class |
|---|---:|---:|
| Training | 60,600 | 600 |
| Validation | 15,150 | 150 |
| Official test | 25,250 | 250 |

The official training partition is divided using stratified sampling with seed 42. RGB normalization statistics are calculated from the training subset only.

The notebook downloads the dataset through `torchvision.datasets.Food101` into `data/raw/`. Dataset files are excluded from Git. Allow approximately 5 GB for the download plus additional space for extraction.

Training uses random resized crops and horizontal flips. Validation and test use deterministic resizing and center cropping. All models receive 128 × 128 RGB inputs.

## Project Files

- [Notebook](deep_learning.ipynb)
- [Analysis report — PDF](reports/Deep_Learning_Systems_Analysis_Report.pdf)
- [Analysis report — Markdown](reports/Deep_Learning_Systems_Analysis_Report.md)
- [Dependencies](requirements.txt)
- [Figures](figures/)
- [Selected checkpoints and training histories](checkpoints/)
- [Saved split indices](splits/)

## Setup and Execution

The recorded environment uses Python 3.14.0, PyTorch 2.14.1, and torchvision 0.29.1 on Apple Silicon with MPS.

```bash
git clone https://github.com/ashlabs/food-recognition-deep-learning.git
cd food-recognition-deep-learning
git checkout develop

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Open `deep_learning.ipynb` from the repository root and run the cells in order. The notebook selects MPS when available and otherwise uses CPU; CUDA selection is not currently implemented.

Pinned dependencies record the original environment but do not guarantee compatibility or identical numerical results on every platform.

## Saved Results and Retraining

When a matching checkpoint and history CSV exist, `run_experiment` loads the saved model and history instead of retraining. The notebook still runs preprocessing checks, a short training pilot, evaluations, and plotting cells.

To train a fresh configuration, use a new `run_name` in its experiment call. Retain the original names to reproduce evaluation using the included checkpoints.

Checkpoints contain selected model weights and metadata, but not optimizer or random states. They support evaluation, not exact continuation of interrupted training.

## Error Analysis and Limitations

Class-level analysis of the wider CNN without dropout found recall ranging from 19.6% for apple pie to 96.8% for edamame. Frequent confusions included hamburger versus pulled pork sandwich and steak versus filet mignon.

Experiments use one seed and one validation partition. Additional experiments were introduced after earlier test results were observed, although every checkpoint was selected using validation loss. Later test results should therefore be interpreted as exploratory benchmark comparisons.

Food-category predictions do not establish ingredients, allergens, portion sizes, or nutritional content. This project is an educational classifier, not a dietary-safety tool.

## Dataset Reference

Bossard, L., Guillaumin, M., & Van Gool, L. (2014).
Food-101 — Mining discriminative components with random forests.
In *Computer Vision — ECCV 2014* (pp. 446–461). Springer.

Additional references and methodological discussion appear in the analysis report.