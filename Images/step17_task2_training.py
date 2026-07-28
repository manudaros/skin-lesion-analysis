"""
Step 17 — Task 2 model definition (dermoscopic attribute detection).

Despite the filename, this file defines the *blueprint* only — matching
the convention of step14_task1_training.py, which is likewise a model
definition rather than a training loop. The Task 2 training loop is
step 18.

Architecture: the same ResNet34 U-Net used for Task 1, with two changes.

  1. Five output channels instead of one — one segmentation map per
     attribute, in the canonical ATTRIBUTES order. The five attributes
     are not mutually exclusive (an image can show both a pigment network
     and globules), so this is five independent binary problems sharing
     an encoder, NOT a 5-class softmax. Every channel gets its own
     sigmoid.

  2. An optional auxiliary classification head hanging off the encoder
     bottleneck, producing one presence logit per attribute. Task 3 needs
     a probability per attribute, and asking "are streaks present?" is a
     far easier question than "which pixels are streaks?" — especially
     for the rare attributes. The head costs almost nothing to train and
     gives Task 3 a cleaner signal than averaging sparse per-pixel logits.

Updated for the new step 15 output layout. Task 1 checkpoint resolution
now lives here rather than being duplicated in step 18 and step 19 — one
place to fix when paths move, which they just did. Both the new
fold-and-resolution directories and the legacy flat location are
searched, so an already-trained checkpoint still resolves.

Run this file directly to self-test the architecture:

    python step17_task2_training.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


# =====================================================================
# 0. Attribute order — the single source of truth
# =====================================================================
# Output channel i corresponds to ATTRIBUTES[i]. Every downstream script
# must use this same list, or channel 3 will mean "globules" in one file
# and "milia-like cysts" in another and nothing will ever line up.

try:
    from qc_config import ATTRIBUTES
except ImportError:
    ATTRIBUTES = [
        "pigment_network",
        "negative_network",
        "streaks",
        "milia_like_cysts",
        "globules",
    ]
    print("qc_config not importable — using the built-in ATTRIBUTES order.")

NUM_ATTRIBUTES = len(ATTRIBUTES)


# =====================================================================
# 1. Where Task 1 checkpoints live
# =====================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TASK1_CHECKPOINT_NAME = "task1_best_model.pth"


def locate_task1_checkpoint(fold: int = 0,
                            image_size: int = 384) -> Optional[Path]:
    """
    Find the best Task 1 checkpoint, or return None.

    Searches the step 15 layout first — one directory per fold and
    resolution, so runs don't overwrite each other — then falls back to
    the old flat location, so a checkpoint trained before the paths
    changed is still picked up.

    Step 18 and step 19 should both call this rather than building paths
    of their own. When step 15 runs all five folds, passing a different
    fold here is all that's needed to warm-start from the matching one.
    """
    candidates = [
        # Current step 15 layout.
        PROJECT_ROOT / "outputs" / "task1_training"
        / f"fold_{fold}_{image_size}px" / TASK1_CHECKPOINT_NAME,
        SCRIPT_DIR / "outputs" / "task1_training"
        / f"fold_{fold}_{image_size}px" / TASK1_CHECKPOINT_NAME,
        # Legacy layout, kept so existing checkpoints still resolve.
        PROJECT_ROOT / "outputs" / "training_results" / TASK1_CHECKPOINT_NAME,
        SCRIPT_DIR / "outputs" / "training_results" / TASK1_CHECKPOINT_NAME,
        Path.cwd() / "outputs" / "training_results" / TASK1_CHECKPOINT_NAME,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


# =====================================================================
# 2. Configuration
# =====================================================================

@dataclass
class Task2ModelConfig:
    """All architecture choices in one place."""

    encoder_name: str = "resnet34"
    encoder_weights: Optional[str] = "imagenet"   # None = random init
    in_channels: int = 3
    classes: int = NUM_ATTRIBUTES

    # Auxiliary presence-classification head.
    use_classification_head: bool = True
    classification_dropout: float = 0.2

    # Warm-start the encoder from a trained Task 1 checkpoint.
    #   task1_checkpoint  — explicit path, wins if set
    #   auto_locate_task1 — otherwise search using fold and image_size
    task1_checkpoint: Optional[str] = None
    auto_locate_task1: bool = False
    fold: int = 0
    image_size: int = 384


# =====================================================================
# 3. The model
# =====================================================================

class Task2Model(nn.Module):
    """
    ResNet34 U-Net with five output channels and an optional presence head.

    forward() returns a tuple:

        seg_logits : [B, 5, H, W]      raw logits, one map per attribute
        cls_logits : [B, 5] or None    raw logits, one presence score each

    Raw logits, not probabilities — apply sigmoid yourself, or better,
    use BCEWithLogitsLoss, which does it internally and more stably.
    """

    def __init__(self, config: Optional[Task2ModelConfig] = None):
        super().__init__()
        self.config = config or Task2ModelConfig()

        aux_params = None
        if self.config.use_classification_head:
            aux_params = {
                "classes": self.config.classes,
                "dropout": self.config.classification_dropout,
                "activation": None,      # keep logits
            }

        self.net = smp.Unet(
            encoder_name=self.config.encoder_name,
            encoder_weights=self.config.encoder_weights,
            in_channels=self.config.in_channels,
            classes=self.config.classes,
            activation=None,             # keep logits
            aux_params=aux_params,
        )

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out = self.net(x)
        if self.config.use_classification_head:
            seg_logits, cls_logits = out
            return seg_logits, cls_logits
        return out, None

    # -- encoder helpers, same idea as step 14 --------------------------

    def freeze_encoder(self) -> None:
        """Train only the decoder. Useful for the first epoch or two."""
        for param in self.net.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.net.encoder.parameters():
            param.requires_grad = True

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =====================================================================
# 4. Warm-starting from the Task 1 encoder
# =====================================================================

def load_task1_encoder(model: Task2Model, checkpoint_path) -> int:
    """
    Copy the encoder weights from your trained Task 1 model into this one.

    Both models are ResNet34 U-Nets over the same 2700 dermoscopy images,
    so the Task 1 encoder has already learned skin texture, hair, colour
    variation and lesion boundaries. Starting from that instead of from
    ImageNet (photographs of dogs and cars) usually converges faster and
    ends up better — and it is a defensible design decision to write up.

    Only the encoder is copied. The decoder must be fresh: Task 1's has
    one output channel, Task 2's has five, so the shapes disagree.

    Implementation note: smp's ResNetEncoder overrides load_state_dict to
    strip the ImageNet classifier weights, and returns None rather than
    the usual (missing, unexpected) pair. So this compares key sets
    directly instead of trusting the return value, and then verifies on a
    sample tensor that the weights genuinely changed — a silent no-op
    would otherwise be indistinguishable from a successful warm start.

    Returns the number of tensors copied.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No Task 1 checkpoint at {checkpoint_path}")

    task1_state = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )

    encoder_state = {
        key[len("encoder."):]: value
        for key, value in task1_state.items()
        if key.startswith("encoder.")
    }

    if not encoder_state:
        raise KeyError(
            "No keys starting with 'encoder.' in the Task 1 checkpoint. "
            "Its state_dict may be nested — inspect it with "
            "list(torch.load(path, weights_only=True).keys())[:5]"
        )

    target_state = model.net.encoder.state_dict()
    target_keys = set(target_state.keys())
    source_keys = set(encoder_state.keys())

    matched = sorted(target_keys & source_keys)
    missing = sorted(target_keys - source_keys)
    unexpected = sorted(source_keys - target_keys)

    # Drop anything whose shape disagrees rather than failing on it.
    mismatched = [
        key for key in matched
        if target_state[key].shape != encoder_state[key].shape
    ]
    for key in mismatched:
        encoder_state.pop(key)

    # Keep one reference tensor so the copy can be verified afterwards.
    probe_key = next(
        (key for key in matched
         if key not in mismatched and target_state[key].dim() > 1),
        None,
    )
    before = target_state[probe_key].clone() if probe_key else None

    model.net.encoder.load_state_dict(encoder_state, strict=False)

    copied = len(matched) - len(mismatched)
    print(f"Warm-started encoder from {checkpoint_path.name}: "
          f"{copied} tensors copied, {len(missing)} missing, "
          f"{len(unexpected)} unexpected, {len(mismatched)} shape mismatch.")
    print(f"  Source: {checkpoint_path}")

    if probe_key is not None:
        after = model.net.encoder.state_dict()[probe_key]
        changed = not torch.allclose(before, after)
        status = ("changed — warm start worked" if changed
                  else "UNCHANGED — warm start did nothing")
        print(f"  Verified on '{probe_key}': weights {status}")

    return copied


# =====================================================================
# 5. Factory
# =====================================================================

def build_task2_model(config: Optional[Task2ModelConfig] = None
                      ) -> Task2Model:
    """Build the Task 2 model. Step 18 and step 19 both call this."""
    config = config or Task2ModelConfig()
    model = Task2Model(config)

    checkpoint = config.task1_checkpoint
    if checkpoint is None and config.auto_locate_task1:
        found = locate_task1_checkpoint(config.fold, config.image_size)
        if found is None:
            print("No Task 1 checkpoint found — "
                  "starting from ImageNet weights.")
        else:
            checkpoint = found

    if checkpoint is not None:
        load_task1_encoder(model, checkpoint)

    return model


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =====================================================================
# 6. Self-test
# =====================================================================

def main() -> None:
    device = select_device()
    print(f"Self-test device: {device}\n")

    print("Attribute order (output channel -> attribute):")
    for index, name in enumerate(ATTRIBUTES):
        print(f"  channel {index} : {name}")
    print()

    found = locate_task1_checkpoint(fold=0, image_size=384)
    print(f"Task 1 checkpoint: {found if found else 'not found'}")

    # The warm start is exercised here so the self-test covers it too.
    config = Task2ModelConfig(auto_locate_task1=True, fold=0, image_size=384)
    model = build_task2_model(config).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters     : {total:,}")
    print(f"Trainable parameters : {model.trainable_parameter_count():,}")
    print(f"Classification head  : {config.use_classification_head}\n")

    # --- forward pass on a dummy batch ---------------------------------
    batch_size, size = 2, 384
    dummy = torch.randn(batch_size, 3, size, size, device=device)

    model.eval()
    with torch.no_grad():
        seg_logits, cls_logits = model(dummy)

    print(f"Input              : {tuple(dummy.shape)}")
    print(f"Segmentation out   : {tuple(seg_logits.shape)}")
    expected_seg = (batch_size, NUM_ATTRIBUTES, size, size)
    assert tuple(seg_logits.shape) == expected_seg, (
        f"Expected {expected_seg}, got {tuple(seg_logits.shape)}"
    )

    if cls_logits is not None:
        print(f"Classification out : {tuple(cls_logits.shape)}")
        assert tuple(cls_logits.shape) == (batch_size, NUM_ATTRIBUTES)

    # Logits should straddle zero, not already be squashed into [0, 1].
    print(f"Logit range        : {seg_logits.min():.3f} "
          f"to {seg_logits.max():.3f}  (raw logits, as expected)\n")

    # --- backward pass: confirm gradients flow -------------------------
    model.train()
    seg_logits, cls_logits = model(dummy)

    target_seg = torch.randint(0, 2, expected_seg, device=device).float()
    loss = nn.functional.binary_cross_entropy_with_logits(
        seg_logits, target_seg
    )

    if cls_logits is not None:
        target_cls = torch.randint(
            0, 2, (batch_size, NUM_ATTRIBUTES), device=device
        ).float()
        loss = loss + 0.5 * nn.functional.binary_cross_entropy_with_logits(
            cls_logits, target_cls
        )

    loss.backward()

    grads = [p.grad.abs().mean().item()
             for p in model.parameters()
             if p.grad is not None]
    print(f"Backward pass OK. Loss {loss.item():.4f}, "
          f"{len(grads)} tensors received gradients.")

    # --- freeze / unfreeze --------------------------------------------
    model.freeze_encoder()
    frozen = model.trainable_parameter_count()
    model.unfreeze_encoder()
    thawed = model.trainable_parameter_count()
    print(f"Encoder frozen     : {frozen:,} trainable")
    print(f"Encoder unfrozen   : {thawed:,} trainable")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
