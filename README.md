# Credit Card OCR - Optical Character Recognition

A deep learning project for recognizing 16-digit credit card numbers from images using a Sequential CNN architecture built with PyTorch.

## Repository Contents

| File | Description |
|------|-------------|
| `ocr_(1).ipynb` | Model code — training, evaluation, and results (Google Colab) |
| `ocr_summary.pdf` | Full technical report — architecture, math, and analysis |
| `dataset_examples/` | Sample images and labels from the synthetic dataset |

## Quick Overview

- **Task**: Predict all 16 digits of a credit card number from a single image
- **Approach**: Fully convolutional backbone that reduces the image to a 1×16 spatial sequence, followed by a shared MLP classifier
- **Result**: **100% sequence-level accuracy** on validation data

For full details on the architecture, loss formulation, and evaluation methodology, see [`ocr_summary.pdf`](ocr_summary.pdf).

## Dataset Examples

The model was trained on 1,200 synthetic credit card images. See [`dataset_examples/`](dataset_examples/) for samples:

| Image | Card Number |
|-------|-------------|
| card_0000.png | `8151 8764 9555 4248` |
| card_0250.png | `6400 7323 9251 8356` |
| card_0500.png | `3773 1542 3047 8490` |

## How to Run

1. Open `ocr_(1).ipynb` in **Google Colab** (GPU recommended)
2. The notebook downloads the dataset automatically from Google Drive
3. Run all cells to train and evaluate

## Technologies

Python · PyTorch · OpenCV · NumPy · Matplotlib · Google Colab (T4 GPU)
