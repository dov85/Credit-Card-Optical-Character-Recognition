# Credit Card OCR - Optical Character Recognition

A deep learning project for recognizing credit card numbers from images using a custom **Multi-Output CNN (MOCNN)** architecture built with PyTorch.

## Overview

This project implements an end-to-end pipeline for credit card number recognition:

1. **Synthetic Data Generation** - Automatically generates realistic credit card images with random 16-digit numbers
2. **Data Augmentation** - Applies perspective transformations to overlay synthetic cards onto real-world background images
3. **Model Training** - Trains a CNN-based model to predict all 16 digits simultaneously
4. **Evaluation** - Achieves **100% accuracy** on the validation set

## Model Architecture

The model (`MOCNN`) uses a multi-output CNN approach:

- **7 CNN layers** with BatchNorm and MaxPooling for feature extraction
- **Embedding layer** (512 → 256 → 128 → 10) for digit classification
- **Multi-head output** - predicts all 16 digit positions simultaneously
- **~5.3M trainable parameters**
- Input: Grayscale image (64×128)
- Output: 16 digits (each classified as 0-9)

## Dataset

The dataset consists of **1,200 synthetic credit card images** with corresponding labels.

- Each image contains a credit card with a randomly generated 16-digit number
- Labels are stored as text files with the format: `XXXX XXXX XXXX XXXX`
- See [dataset_examples/](dataset_examples/) for sample images and labels

### Dataset Examples

| Image | Card Number |
|-------|-------------|
| card_0000.png | `8151 8764 9555 4248` |
| card_0250.png | `6400 7323 9251 8356` |
| card_0500.png | `3773 1542 3047 8490` |
| card_0750.png | `8254 8162 2500 7642` |
| card_0999.png | `6379 4076 3038 9277` |

## Training Results

- **Epochs**: 30
- **Batch Size**: 30
- **Learning Rate**: 0.0001
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Final Train Loss**: 0.0001
- **Final Val Loss**: 0.0003
- **Validation Accuracy**: 100%

## Project Structure

```
├── ocr_(1).ipynb          # Main notebook - model training & evaluation
├── dataset_examples/      # Sample data for demonstration
│   ├── images/            # 5 sample credit card images
│   └── labels/            # Corresponding ground-truth labels
└── README.md
```

## How to Run

1. Open `ocr_(1).ipynb` in Google Colab (GPU recommended)
2. The notebook will automatically download the dataset from Google Drive
3. Run all cells to train the model and evaluate results

## Technologies

- Python 3
- PyTorch
- OpenCV
- PIL / Pillow
- NumPy
- Matplotlib
- Google Colab (T4 GPU)

## License

This project is for educational purposes.
