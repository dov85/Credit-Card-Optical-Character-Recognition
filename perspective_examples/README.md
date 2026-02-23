# Perspective Dataset Examples

This folder contains **5 sample triplets** from the full perspective credit card dataset (1,200 samples total).

## Structure

```
perspective_examples/
├── images/         # Background images with synthetic card overlaid (.jpg)
├── matrices/       # 3×3 perspective straightening matrices (.npy)
├── quads/          # 4-point quad coordinates of card corners (.json)
└── labels/         # 16-digit card number ground truth (.txt)
```

## Sample Data

| File | Card Number | Matrix M[0,2] (tx) | Matrix M[1,2] (ty) |
|------|-------------|---------------------|---------------------|
| sample_0000 | `8151 8764 9555 4248` | -31.3 | -352.6 |
| sample_0005 | `9433 8445 3365 2029` | -405.1 | -1373.4 |
| sample_0010 | `3442 3913 4340 4354` | — | — |
| sample_0015 | `8878 2707 6105 1047` | — | — |
| sample_0020 | `8222 1655 4803 8030` | — | — |

## File Formats

### Matrix (`.npy`)
A 3×3 `float32` NumPy array — the perspective transform that maps the warped card quad to a straight rectangle:

```python
import numpy as np
M = np.load("matrices/sample_0000.npy")
# [[ 4.00e-01, -1.01e-01, -3.13e+01],
#  [ 9.98e-02,  4.18e-01, -3.53e+02],
#  [-2.38e-05,  2.95e-05,  1.00e+00]]
```

### Quad (`.json`)
The 4 corner points of the card in the image, in pixel coordinates:

```json
{"quad": [[273.8, 778.3], [1453.7, 496.5], [1655.7, 1216.1], [462.9, 1530.0]]}
```

### Label (`.txt`)
The 16-digit card number as 4 groups of 4 digits:
```
8151 8764 9555 4248
```
