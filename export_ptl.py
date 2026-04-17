# Copyright 2025 Pierre-Yves Nicolas

# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""
Export a PyTorch Mobile (.ptl) int8-quantized segmentation model.

Run *after* train.py has produced build/model/fairscan-segmentation-model.pt
and left the validation images under build/dataset/fairscan-dataset/val/.

Strategy:
  - FX graph-mode post-training static quantization targeting QNNPACK.
  - Bilinear upsample ops (ASPP pool branch + SegmentationHead 4x upsample)
    are intentionally kept in fp32: QNNPACK's quantized bilinear upsample
    collapses activations to a near-constant output, which is the usual
    cause of a "quantized model detects nothing" failure on this
    architecture.
  - Calibration runs on real validation images so activation ranges match
    production inputs.
"""

import glob
import os

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.ao.quantization import QConfigMapping, get_default_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.utils.mobile_optimizer import optimize_for_mobile

BUILD_DIR = "build"
MODEL_DIR = BUILD_DIR + "/model"
FP32_MODEL_PATH = MODEL_DIR + "/fairscan-segmentation-model.pt"
PTL_MODEL_PATH = MODEL_DIR + "/fairscan-segmentation-model.ptl"
VAL_IMAGE_DIR = BUILD_DIR + "/dataset/fairscan-dataset/val/images"
VAL_MASK_DIR = BUILD_DIR + "/dataset/fairscan-dataset/val/masks"

ENCODER = "mobilenet_v2"
INPUT_SIZE = 256
CALIBRATION_IMAGES = 100

torch.backends.quantized.engine = "qnnpack"


class TraceFriendlyDeepLabV3Plus(nn.Module):
    """Strips SegmentationModel's dynamic shape assertions so FX can trace it."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.encoder = base.encoder
        self.decoder = base.decoder
        self.segmentation_head = base.segmentation_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        return self.segmentation_head(decoder_output)


def load_fp32_model() -> nn.Module:
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load(FP32_MODEL_PATH, map_location="cpu"))
    model.eval()
    return TraceFriendlyDeepLabV3Plus(model).eval()


def build_qconfig_mapping() -> QConfigMapping:
    qconfig = get_default_qconfig("qnnpack")
    # Bilinear upsample on QNNPACK is unstable. Leave every upsample op in
    # fp32; FX inserts DeQuant/Quant around the fp32 island automatically.
    return (
        QConfigMapping()
        .set_global(qconfig)
        .set_object_type(F.interpolate, None)
        .set_object_type(nn.Upsample, None)
        .set_object_type(nn.UpsamplingBilinear2d, None)
        .set_object_type(nn.UpsamplingNearest2d, None)
    )


def calibration_transform() -> A.Compose:
    return A.Compose([
        A.Resize(INPUT_SIZE, INPUT_SIZE),
        A.Normalize(),  # ImageNet mean/std. Android preprocessing MUST match.
        ToTensorV2(),
    ])


def iter_calibration_batches(transform: A.Compose, limit: int):
    paths = sorted(glob.glob(os.path.join(VAL_IMAGE_DIR, "*.jpg")))[:limit]
    if not paths:
        raise RuntimeError(
            f"No calibration images under {VAL_IMAGE_DIR}. Run train.py first."
        )
    for p in paths:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        dummy_mask = np.zeros(img.shape[:2], dtype=np.float32)
        tensor = transform(image=img, mask=dummy_mask)["image"]
        yield tensor.unsqueeze(0)


def dice_score(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum()
    return ((2 * inter + 1e-6) / (pred.sum() + target.sum() + 1e-6)).item()


def compare_dice(fp32: nn.Module, int8: nn.Module, transform: A.Compose, limit: int = 30):
    img_paths = sorted(glob.glob(os.path.join(VAL_IMAGE_DIR, "*.jpg")))[:limit]
    fp32_scores, int8_scores = [], []
    with torch.no_grad():
        for p in img_paths:
            name = os.path.splitext(os.path.basename(p))[0]
            mpath = os.path.join(VAL_MASK_DIR, name + ".png")
            if not os.path.exists(mpath):
                continue
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            mask = (cv2.imread(mpath, cv2.IMREAD_GRAYSCALE) > 127).astype("float32")
            aug = transform(image=img, mask=mask)
            x = aug["image"].unsqueeze(0)
            y = aug["mask"].unsqueeze(0).unsqueeze(0)
            fp32_scores.append(dice_score(fp32(x), y))
            int8_scores.append(dice_score(int8(x), y))
    return fp32_scores, int8_scores


def main():
    print("Loading fp32 model...")
    fp32_model = load_fp32_model()

    print("Preparing FX graph-mode quantization (QNNPACK, upsample left fp32)...")
    example_input = (torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE),)
    prepared = prepare_fx(fp32_model, build_qconfig_mapping(), example_input)

    transform = calibration_transform()
    print(f"Calibrating on up to {CALIBRATION_IMAGES} validation images...")
    with torch.no_grad():
        for i, x in enumerate(iter_calibration_batches(transform, CALIBRATION_IMAGES)):
            prepared(x)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1} images")

    print("Converting to int8...")
    quantized = convert_fx(prepared).eval()

    print("Sanity-check: fp32 vs int8 Dice on validation split")
    fp32_scores, int8_scores = compare_dice(fp32_model, quantized, transform)
    if fp32_scores:
        print(f"  fp32 Dice: {np.mean(fp32_scores):.4f}")
        print(f"  int8 Dice: {np.mean(int8_scores):.4f}")
        if np.mean(int8_scores) < 0.5 * np.mean(fp32_scores):
            print("  WARNING: int8 Dice dropped >50%. Re-check preprocessing.")

    print("Tracing + optimize_for_mobile + save for lite interpreter...")
    traced = torch.jit.trace(quantized, example_input[0])
    optimized = optimize_for_mobile(traced)
    optimized._save_for_lite_interpreter(PTL_MODEL_PATH)

    size_mb = os.path.getsize(PTL_MODEL_PATH) / 1e6
    print(f"\nWrote {PTL_MODEL_PATH} ({size_mb:.2f} MB)")
    print("Android-side contract:")
    print("  input: FloatTensor shape (1, 3, 256, 256), NCHW")
    print("  preprocessing: resize to 256x256, then normalize with")
    print("    mean = [0.485, 0.456, 0.406]")
    print("    std  = [0.229, 0.224, 0.225]")
    print("  output: FloatTensor shape (1, 1, 256, 256), raw logits")
    print("          apply sigmoid + threshold 0.5 to get binary mask")


if __name__ == "__main__":
    main()
