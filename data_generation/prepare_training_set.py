"""
Pre-resize the perspective dataset for training.

The raw images are 2160x3840 (~0.7 MB each, ~885 MB total). The model only ever sees them
at 256x256, so decoding full-resolution JPEGs on every epoch wastes far more time than the
forward pass itself — and a 779 MB zip is painful to move to Colab.

This script resizes once, producing a folder of roughly 30 MB that trains identically.

The original image dimensions are written to `orig_sizes.json`, because the ground-truth
homographies are expressed in original-resolution pixel coordinates: the transformer needs
the original size to normalise correctly, and it can no longer read it off the image file.

    python data_generation/prepare_training_set.py
    # then zip the output folder and upload it to Drive
"""

import json
import os
import shutil
import sys

from PIL import Image

SRC = "perspective_dataset_with_cards"
DST = "perspective_dataset_768"

# Must match IMG_SIZE in perspective_ocr_model.ipynb.
#
# 256 was the first choice and it is enough to locate the card — corner error reached 1.7 px
# with it. It is not enough to *read* the card: at 256 the card is about 149 px wide, so a
# single digit is 5.6 px, and per-digit accuracy stalled at 85%. Resolution the resize threw
# away cannot be recovered by any later upsampling.
#
# At 768 a digit is about 17 px. The corner network still sees a 256 downscale, so only the
# sampling of the crop gets the extra detail.
SIZE = (768, 768)


def main(src=SRC, dst=DST, size=SIZE):
    img_src = os.path.join(src, "images")
    if not os.path.isdir(img_src):
        sys.exit(f"Not found: {img_src}\nRun this from the repository root.")

    for sub in ("images", "matrices", "labels"):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)

    files = sorted(f for f in os.listdir(img_src) if f.endswith((".jpg", ".png")))
    orig_sizes = {}

    for n, fname in enumerate(files, 1):
        stem = os.path.splitext(fname)[0]

        with Image.open(os.path.join(img_src, fname)) as im:
            orig_sizes[stem] = list(im.size)          # (width, height) before resizing
            im.convert("RGB").resize(size, Image.BILINEAR).save(
                os.path.join(dst, "images", f"{stem}.jpg"), quality=95)

        for sub, ext in (("matrices", ".npy"), ("labels", ".txt")):
            source = os.path.join(src, sub, stem + ext)
            if os.path.exists(source):
                shutil.copy2(source, os.path.join(dst, sub, stem + ext))

        if n % 200 == 0 or n == len(files):
            print(f"  {n}/{len(files)}")

    with open(os.path.join(dst, "orig_sizes.json"), "w") as f:
        json.dump(orig_sizes, f)

    before = sum(os.path.getsize(os.path.join(img_src, f)) for f in files)
    after = sum(os.path.getsize(os.path.join(dst, "images", f))
                for f in os.listdir(os.path.join(dst, "images")))
    print(f"\nDone: {len(files)} samples -> {dst}/")
    print(f"Images {before / 1e6:.0f} MB -> {after / 1e6:.0f} MB "
          f"({before / max(after, 1):.0f}x smaller)")


if __name__ == "__main__":
    main()
