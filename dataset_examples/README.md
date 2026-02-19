# Dataset Examples

This folder contains **5 sample pairs** from the full synthetic credit card dataset (1,200 images total).

## Structure

```
dataset_examples/
├── images/          # Synthetic credit card images (.png)
│   ├── card_0000.png
│   ├── card_0250.png
│   ├── card_0500.png
│   ├── card_0750.png
│   └── card_0999.png
└── labels/          # Ground-truth card numbers (.txt)
    ├── card_0000.txt
    ├── card_0250.txt
    ├── card_0500.txt
    ├── card_0750.txt
    └── card_0999.txt
```

## Label Format

Each `.txt` file contains the 16-digit card number as 4 groups of 4 digits separated by spaces:

| File | Label |
|------|-------|
| card_0000.txt | `8151 8764 9555 4248` |
| card_0250.txt | `6400 7323 9251 8356` |
| card_0500.txt | `3773 1542 3047 8490` |
| card_0750.txt | `8254 8162 2500 7642` |
| card_0999.txt | `6379 4076 3038 9277` |

## Full Dataset

The full dataset contains 1,200 image-label pairs and can be regenerated using `card_fix.py`:

```bash
python card_fix.py
```

The augmented dataset (with perspective-warped cards on real backgrounds) can be generated using `creat.py`.
