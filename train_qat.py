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
QAT fine-tune and .ptl export.

Run *after* train.py has produced build/model/fairscan-segmentation-model.pt.
This script loads the fp32 checkpoint, inserts FakeQuantize into the encoder
via prepare_qat_fx, fine-tunes for a few epochs so weights adapt to int8
noise, then converts and exports a lite-interpreter .ptl.

Why QAT instead of PTQ:
  Post-training static quantization of DeepLabV3Plus + MobileNetV2 on QNNPACK
  collapses the output to near-constant noise regardless of per-channel /
  histogram / selective-quant tweaks. Depthwise conv channel-wise weight
  distribution and BN running stats cannot be "calibrated" after the fact.
  QAT makes the weights adapt to quantization noise during training.
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
import torchvision.transforms.functional as TF
from albumentations.pytorch import ToTensorV2
from torch.ao.quantization import QConfig
from torch.ao.quantization.backend_config import get_qnnpack_backend_config
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx
from torch.utils.data import DataLoader, Dataset
from torch.utils.mobile_optimizer import optimize_for_mobile

BUILD_DIR = "build"
MODEL_DIR = BUILD_DIR + "/model"
FP32_MODEL_PATH = MODEL_DIR + "/fairscan-segmentation-model.pt"
PTL_MODEL_PATH = MODEL_DIR + "/fairscan-segmentation-model.ptl"
DATASET_DIR = BUILD_DIR + "/dataset/fairscan-dataset"
TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, "train/images")
TRAIN_MASK_DIR = os.path.join(DATASET_DIR, "train/masks")
VAL_IMAGE_DIR = os.path.join(DATASET_DIR, "val/images")
VAL_MASK_DIR = os.path.join(DATASET_DIR, "val/masks")

ENCODER = "mobilenet_v2"
INPUT_SIZE = 256
QAT_EPOCHS = 3
QAT_BATCH_SIZE = 16
QAT_NUM_WORKERS = 4
QAT_LR = 1e-5  # fine-tune LR, 20x smaller than the 1e-4 used in train.py
FREEZE_BN_AFTER_EPOCH = 1  # BN running stats frozen from epoch 2 onward
FREEZE_OBSERVER_AFTER_EPOCH = 1  # activation observers frozen from epoch 2 onward

torch.backends.quantized.engine = "qnnpack"


class DocumentSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype("float32")
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"].unsqueeze(0)
        else:
            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)
        return image, mask

    def __len__(self):
        return len(self.image_paths)


class TraceFriendlyDeepLabV3Plus(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.encoder = base.encoder
        self.decoder = base.decoder
        self.segmentation_head = base.segmentation_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        return self.segmentation_head(decoder_output)


def build_qat_qconfig_mapping() -> QConfigMapping:
    # FakeQuantize with moving-average observers + per-channel symmetric weights.
    # This matches the PTQ qconfig structure we settled on, so the QAT-to-int8
    # conversion produces a model that matches the QNNPACK runtime expectations.
    qat_qconfig = QConfig(
        activation=FusedMovingAvgObsFakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,
            reduce_range=False,
        ),
        weight=FusedMovingAvgObsFakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
        ),
    )
    return (
        QConfigMapping()
        .set_global(None)
        .set_module_name("encoder", qat_qconfig)
        .set_module_name("encoder.features.0", None)
        .set_object_type(F.interpolate, None)
        .set_object_type(nn.Upsample, None)
        .set_object_type(nn.UpsamplingBilinear2d, None)
        .set_object_type(nn.UpsamplingNearest2d, None)
    )


def dice_continuous(pred, target, smooth=1e-6):
    p = pred.contiguous().view(-1)
    t = target.contiguous().view(-1)
    inter = (p * t).sum()
    return (2 * inter + smooth) / (p.sum() + t.sum() + smooth)


def dice_discrete(pred, target, smooth=1e-6):
    return dice_continuous((pred > 0.5).float(), target, smooth)


def evaluate_dice(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            probs = torch.sigmoid(model(images))
            total += dice_discrete(probs, masks).item()
    return total / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading fp32 checkpoint...")
    base = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    base.load_state_dict(torch.load(FP32_MODEL_PATH, map_location="cpu"))
    model = TraceFriendlyDeepLabV3Plus(base)

    shared_transform = A.Compose([
        A.Resize(INPUT_SIZE, INPUT_SIZE),
        A.Normalize(),
        ToTensorV2(),
    ])
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        shared_transform,
    ])
    pin = device.type == "cuda"
    train_loader = DataLoader(
        DocumentSegmentationDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, train_transform),
        batch_size=QAT_BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=QAT_NUM_WORKERS, pin_memory=pin, persistent_workers=QAT_NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        DocumentSegmentationDataset(VAL_IMAGE_DIR, VAL_MASK_DIR, shared_transform),
        batch_size=QAT_BATCH_SIZE, shuffle=False,
        num_workers=QAT_NUM_WORKERS, pin_memory=pin, persistent_workers=QAT_NUM_WORKERS > 0,
    )

    # prepare_qat_fx handles Conv-BN(-ReLU) fusion into QAT-aware modules
    # internally; do NOT pre-fuse with fuse_fx (that's for PTQ only).
    backend_config = get_qnnpack_backend_config()
    model.train()
    example_input = (torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE),)
    prepared = prepare_qat_fx(
        model, build_qat_qconfig_mapping(), example_input, backend_config=backend_config,
    ).to(device)

    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss = nn.BCEWithLogitsLoss()
    def loss_fn(logits, target):
        return dice_loss(logits, target) + bce_loss(logits, target)
    optimizer = torch.optim.Adam(prepared.parameters(), lr=QAT_LR)

    for epoch in range(QAT_EPOCHS):
        prepared.train()
        if epoch > FREEZE_BN_AFTER_EPOCH:
            # Freeze BN running stats: the simulated int8 backward has already
            # shaped the fp32 stats; further updates add noise to the scales.
            prepared.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
        if epoch > FREEZE_OBSERVER_AFTER_EPOCH:
            prepared.apply(torch.ao.quantization.disable_observer)

        total_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = prepared(images)
            loss = loss_fn(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_avg = total_loss / len(train_loader)

        # Skip FakeQuantize eval on intermediate epochs -- eval is as slow as
        # one extra training pass because FakeQuantize runs on forward too.
        if epoch == QAT_EPOCHS - 1:
            fakeq_dice = evaluate_dice(prepared, val_loader, device)
            print(f"[QAT {epoch + 1}/{QAT_EPOCHS}] train_loss={train_avg:.4f} "
                  f"fakeq_dice={fakeq_dice:.4f}", flush=True)
        else:
            print(f"[QAT {epoch + 1}/{QAT_EPOCHS}] train_loss={train_avg:.4f}",
                  flush=True)

    print("Converting QAT model to real int8...")
    prepared.eval().cpu()
    quantized = convert_fx(prepared, backend_config=backend_config).eval()

    int8_dice = evaluate_dice(quantized, val_loader, torch.device("cpu"))
    print(f"\nFinal int8 Dice (real quantized): {int8_dice:.4f}")

    print("Diagnostic: int8 output variability across 3 images")
    paths = sorted(glob.glob(os.path.join(VAL_IMAGE_DIR, "*.jpg")))[:3]
    calib_transform = shared_transform
    with torch.no_grad():
        for p in paths:
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            x = calib_transform(
                image=img, mask=np.zeros(img.shape[:2], dtype=np.float32),
            )["image"].unsqueeze(0)
            out = quantized(x)
            print(f"  {os.path.basename(p)}: "
                  f"mean={out.mean().item():.3f} std={out.std().item():.3f} "
                  f"min={out.min().item():.3f} max={out.max().item():.3f}")

    traced = torch.jit.trace(quantized, example_input[0])
    optimized = optimize_for_mobile(traced)
    optimized._save_for_lite_interpreter(PTL_MODEL_PATH)
    size_mb = os.path.getsize(PTL_MODEL_PATH) / 1e6
    print(f"\nWrote {PTL_MODEL_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
