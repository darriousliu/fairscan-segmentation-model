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
Export an ExecuTorch (.pte) int8-quantized segmentation model via the
PT2 Export + XNNPACK quantization pipeline.

This is an alternative to train_qat.py for apps willing to adopt the
ExecuTorch Android runtime. PT2E + XNNPACKQuantizer handles
DeepLabV3Plus + MobileNetV2 quantization cleanly without QAT:
  - per-channel symmetric weights by default (critical for depthwise convs)
  - Conv-BN folded during prepare_pt2e
  - hard-to-quantize ops (bilinear interpolate) auto-fallback to fp32 via
    the XnnpackPartitioner, no manual op-type exclusion needed

This is the same pipeline style ai-edge-torch (litert-torch) uses to
produce the .tflite in train.py, so near-lossless quantization is
expected.

Tradeoff: requires the ExecuTorch Android runtime alongside any
existing pytorch_android runtime. Ship both only if APK size allows;
otherwise either migrate every model to ExecuTorch or stay on the
train_qat.py .ptl path.

Prerequisites:
  pip install executorch
Run *after* train.py has produced build/model/fairscan-segmentation-model.pt.
"""

import glob
import os

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from executorch.exir import to_edge_transform_and_lower

BUILD_DIR = "build"
MODEL_DIR = BUILD_DIR + "/model"
FP32_MODEL_PATH = MODEL_DIR + "/fairscan-segmentation-model.pt"
PTE_MODEL_PATH = MODEL_DIR + "/fairscan-segmentation-model.pte"
VAL_IMAGE_DIR = BUILD_DIR + "/dataset/fairscan-dataset/val/images"
VAL_MASK_DIR = BUILD_DIR + "/dataset/fairscan-dataset/val/masks"

ENCODER = "mobilenet_v2"
INPUT_SIZE = 256
CALIBRATION_IMAGES = 100


class TraceFriendlyDeepLabV3Plus(nn.Module):
    """Strips SegmentationModel's dynamic shape assertions so torch.export can trace it."""

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


def calibration_transform() -> A.Compose:
    return A.Compose([
        A.Resize(INPUT_SIZE, INPUT_SIZE),
        A.Normalize(),  # ImageNet mean/std -- Android preprocessing MUST match.
        ToTensorV2(),
    ])


def iter_calibration_inputs(transform: A.Compose, limit: int):
    paths = sorted(glob.glob(os.path.join(VAL_IMAGE_DIR, "*.jpg")))[:limit]
    if not paths:
        raise RuntimeError(
            f"No calibration images under {VAL_IMAGE_DIR}. Run train.py first."
        )
    for p in paths:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        tensor = transform(
            image=img, mask=np.zeros(img.shape[:2], dtype=np.float32)
        )["image"]
        yield tensor.unsqueeze(0)


def dice_score(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum()
    return ((2 * inter + 1e-6) / (pred.sum() + target.sum() + 1e-6)).item()


def compare_dice(fp32_model: nn.Module, int8_model, transform: A.Compose, limit: int = 30):
    paths = sorted(glob.glob(os.path.join(VAL_IMAGE_DIR, "*.jpg")))[:limit]
    fp32_scores, int8_scores = [], []
    with torch.no_grad():
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            mpath = os.path.join(VAL_MASK_DIR, name + ".png")
            if not os.path.exists(mpath):
                continue
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            mask = (cv2.imread(mpath, cv2.IMREAD_GRAYSCALE) > 127).astype("float32")
            aug = transform(image=img, mask=mask)
            x = aug["image"].unsqueeze(0)
            y = aug["mask"].unsqueeze(0).unsqueeze(0)
            fp32_scores.append(dice_score(fp32_model(x), y))
            int8_scores.append(dice_score(int8_model(x), y))
    return fp32_scores, int8_scores


def main():
    print("Loading fp32 model...")
    fp32_model = load_fp32_model()
    example_input = (torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE),)

    print("Exporting via torch.export.export_for_training (PT2E entry point)...")
    exported_for_quant = torch.export.export_for_training(
        fp32_model, example_input
    ).module()

    print("Configuring XNNPACKQuantizer (per-channel symmetric int8)...")
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_qat=False)
    )

    print("prepare_pt2e (inserts observers + folds Conv-BN)...")
    prepared = prepare_pt2e(exported_for_quant, quantizer)

    print(f"Calibrating on up to {CALIBRATION_IMAGES} validation images...")
    transform = calibration_transform()
    with torch.no_grad():
        for i, x in enumerate(iter_calibration_inputs(transform, CALIBRATION_IMAGES)):
            prepared(x)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1} images")

    print("convert_pt2e (observers -> real int8 ops)...")
    quantized = convert_pt2e(prepared)

    print("Sanity-check: fp32 vs int8 Dice on validation split")
    fp32_scores, int8_scores = compare_dice(fp32_model, quantized, transform)
    if fp32_scores:
        print(f"  fp32 Dice: {np.mean(fp32_scores):.4f}")
        print(f"  int8 Dice: {np.mean(int8_scores):.4f}")

    print("Diagnostic: int8 output variability across 3 images")
    sample_paths = sorted(glob.glob(os.path.join(VAL_IMAGE_DIR, "*.jpg")))[:3]
    with torch.no_grad():
        for p in sample_paths:
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            x = transform(
                image=img, mask=np.zeros(img.shape[:2], dtype=np.float32),
            )["image"].unsqueeze(0)
            out = quantized(x)
            print(f"  {os.path.basename(p)}: "
                  f"mean={out.mean().item():.3f} std={out.std().item():.3f} "
                  f"min={out.min().item():.3f} max={out.max().item():.3f}")

    print("Lowering to ExecuTorch via XnnpackPartitioner...")
    # Re-export so the ExecuTorch backend sees the int8 graph.
    exported_for_lowering = torch.export.export(quantized, example_input)
    edge = to_edge_transform_and_lower(
        exported_for_lowering,
        partitioner=[XnnpackPartitioner()],
    )
    et_program = edge.to_executorch()

    with open(PTE_MODEL_PATH, "wb") as f:
        f.write(et_program.buffer)

    size_mb = os.path.getsize(PTE_MODEL_PATH) / 1e6
    print(f"\nWrote {PTE_MODEL_PATH} ({size_mb:.2f} MB)")
    print("Android-side contract:")
    print("  runtime: executorch_android .aar (XNNPACK backend delegate)")
    print("  input: FloatTensor shape (1, 3, 256, 256), NCHW")
    print("  preprocessing: resize to 256x256, ImageNet mean/std")
    print("    mean = [0.485, 0.456, 0.406]")
    print("    std  = [0.229, 0.224, 0.225]")
    print("  output: FloatTensor (1, 1, 256, 256) raw logits")
    print("          apply sigmoid + threshold 0.5 for binary mask")


if __name__ == "__main__":
    main()
