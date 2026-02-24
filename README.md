# Credit Card OCR - Optical Character Recognition

A deep learning project for recognizing 16-digit credit card numbers from images using a Sequential CNN architecture built with PyTorch. Includes a full **synthetic data generation pipeline** for creating perspective-warped card images with ground-truth matrices, quad coordinates, and labels.

## Repository Contents

| File / Folder | Description |
|------|-------------|
| `credit_card_ocr_model.ipynb` | Model code — training, evaluation, and results (Google Colab) |
| `ocr_summary.pdf` | Full technical report — architecture, math, and analysis |
| `data_generation/` | All data generation scripts (see below) |
| `dataset_examples/` | Sample images and labels from the flat card dataset |
| `perspective_examples/` | Sample images, matrices, quads, and labels from the perspective dataset |

### `data_generation/` Scripts

| Script | Description |
|--------|-------------|
| `generate_cards.py` | **Stage 1** — Generate 1,200 synthetic credit card images with random numbers |
| `generate_perspectives.py` | **Stage 2** — Generate perspective-warped backgrounds with ground-truth matrices |
| `overlay_cards.py` | **Stage 3** — Overlay synthetic cards onto perspective backgrounds |
| `augment_direct.py` | Alternative augmentation — overlay cards directly onto raw backgrounds (no perspective preprocessing) |
| `extract_card.py` | Utility — Extract card region from an image using quad coordinates |
| `generate_perspective_data.ipynb` | Notebook version of Stage 2 for interactive exploration |

---

## Data Generation Pipeline

The dataset is built in 3 stages. Each stage produces data that feeds into the next.

### Stage 1: Synthetic Card Generation (`data_generation/generate_cards.py`)

Generates 1,200 flat credit card images with random 16-digit numbers, expiry dates, and cardholder names.

```
Input:  card.jpeg (template image)
Output: dataset/images/card_XXXX.png    (1,200 card images)
        dataset/labels/card_XXXX.txt    (16-digit card number per image)
```

**Process:**
1. Loads a blank card template (`card.jpeg`)
2. Generates random card number (4 groups × 4 digits), expiry date, ID, and name
3. Renders text onto the card using PIL (Pillow)
4. Crops tightly around the card content using OpenCV contour detection
5. Saves image + label (the 16 digits separated by spaces)

```bash
python data_generation/generate_cards.py
```

### Stage 2: Perspective Background Generation (`data_generation/generate_perspectives.py`)

Creates perspective-warped background scenes with ground-truth transformation matrices. Uses real card-holding photos from `temp_extract/` as source backgrounds.

```
Input:  temp_extract/images/{DG,DX,LG,LX}/*.jpg   (background photos)
        temp_extract/ground_truth/{DG,DX,LG,LX}/*.json  (quad annotations)
Output: perspective_dataset/images/sample_XXXX.jpg      (warped backgrounds)
        perspective_dataset/matrices/sample_XXXX.npy     (3×3 perspective matrix)
        perspective_dataset/quads/sample_XXXX.json       (4-point card corners)
```

**Process:**
1. Loads background images and their quad annotations (4 corners of the card region)
2. Applies random affine transformations (rotation ±30°, scale 0.5–1.2, translation ±20%)
3. Transforms the quad coordinates through the same affine matrix
4. Validates that the transformed quad stays within image bounds
5. Computes the perspective straightening matrix: `M = getPerspectiveTransform(quad → rectangle)`
6. Saves the warped image, the 3×3 matrix (`.npy`), and the transformed quad (`.json`)

```bash
python data_generation/generate_perspectives.py
```

### Stage 3: Card Overlay (`data_generation/overlay_cards.py`)

Combines Stage 1 and Stage 2: overlays each synthetic card onto a random perspective background at the exact quad position.

```
Input:  dataset/images/card_XXXX.png                    (flat cards from Stage 1)
        perspective_dataset/images/sample_XXXX.jpg       (backgrounds from Stage 2)
        perspective_dataset/quads/sample_XXXX.json       (quad coordinates)
        perspective_dataset/matrices/sample_XXXX.npy     (perspective matrices)
Output: perspective_dataset_with_cards/images/sample_XXXX.jpg   (final composites)
        perspective_dataset_with_cards/matrices/sample_XXXX.npy  (unchanged matrices)
        perspective_dataset_with_cards/quads/sample_XXXX.json    (unchanged quads)
        perspective_dataset_with_cards/labels/sample_XXXX.txt    (card number labels)
```

**Process:**
1. For each of 1,200 synthetic cards, picks a random perspective background
2. Computes `getPerspectiveTransform(card_corners → quad)` to warp the flat card into position
3. Creates a binary mask and warps it with the same matrix
4. Blends the warped card onto the background using the mask (bitwise operations)
5. Copies the perspective matrix and quad (unchanged — the card sits exactly in the quad region)
6. Copies the card number label

```bash
python data_generation/overlay_cards.py
```

### Pipeline Diagram

```
┌──────────────────────┐     ┌──────────────────────┐     ┌─────────────────────────┐
│  generate_cards.py    │     │ generate_perspectives │     │ overlay_cards.py         │
│                      │     │ .py                   │     │                         │
│  card.jpeg ──►       │     │                      │     │                         │
│  1,200 flat cards    │────►│ temp_extract/ ──►     │────►│  Card + Background ──►  │
│  + labels            │     │ warped backgrounds    │     │  Final composites       │
│                      │     │ + matrices + quads    │     │  + matrices + labels    │
└──────────────────────┘     └──────────────────────┘     └─────────────────────────┘
       Stage 1                      Stage 2                       Stage 3
```

---

## Ground Truth Format

Each sample in the final dataset contains 4 files:

| File | Format | Description |
|------|--------|-------------|
| `sample_XXXX.jpg` | JPEG image | Background with card overlaid at perspective angle |
| `sample_XXXX.npy` | NumPy 3×3 float32 | Perspective matrix that straightens the card |
| `sample_XXXX.json` | JSON with `"quad"` key | 4 corner points of the card in pixel coordinates |
| `sample_XXXX.txt` | Text | 16-digit card number (e.g., `8151 8764 9555 4248`) |

### Perspective Matrix

The 3×3 matrix maps from the warped card quad to a straight rectangle:

```python
import numpy as np, cv2
M = np.load("matrices/sample_0000.npy")
# [[ 4.00e-01, -1.01e-01, -3.13e+01],   ← rotation + translation
#  [ 9.98e-02,  4.18e-01, -3.53e+02],   ← rotation + translation
#  [-2.38e-05,  2.95e-05,  1.00e+00]]   ← perspective distortion

straightened = cv2.warpPerspective(image, M, (512, 323))
```

Element ranges across the full dataset:
- **M[0,0], M[0,1], M[1,0], M[1,1]** (rotation/scale): ~0.3 – 1.0
- **M[0,2], M[1,2]** (translation): ~-1400 – 500
- **M[2,0], M[2,1]** (projective): ~-5e-4 – 5e-4
- **M[2,2]**: always 1.0

---

## Model

### Flat Card OCR (`credit_card_ocr_model.ipynb`)

- **Task**: Predict 16 digits from a flat (non-perspective) card image
- **Architecture**: Sequential CNN (MOCNN) — conv backbone reduces to 1×16 spatial sequence → shared MLP classifier
- **Result**: **100% sequence-level accuracy** on validation data

For full details on the architecture, loss formulation, and evaluation methodology, see [`ocr_summary.pdf`](ocr_summary.pdf).

---

## Dataset Examples

### Flat Cards ([`dataset_examples/`](dataset_examples/))

| Image | Card Number |
|-------|-------------|
| card_0000.png | `8151 8764 9555 4248` |
| card_0250.png | `6400 7323 9251 8356` |
| card_0500.png | `3773 1542 3047 8490` |

### Perspective Cards ([`perspective_examples/`](perspective_examples/))

5 sample triplets (image + matrix + quad + label) from the 1,200-sample perspective dataset. See [`perspective_examples/README.md`](perspective_examples/README.md) for details.

---

## How to Run

### Option A: Flat Card OCR
1. Open `credit_card_ocr_model.ipynb` in **Google Colab** (GPU recommended)
2. The notebook downloads the dataset automatically from Google Drive
3. Run all cells to train and evaluate

### Option B: Regenerate Data Locally
```bash
python data_generation/generate_cards.py          # Stage 1: 1,200 flat cards
python data_generation/generate_perspectives.py   # Stage 2: perspective backgrounds
python data_generation/overlay_cards.py            # Stage 3: overlay cards
```

## Technologies

Python · PyTorch · OpenCV · NumPy · PIL/Pillow · Matplotlib · Google Colab (T4 GPU)
