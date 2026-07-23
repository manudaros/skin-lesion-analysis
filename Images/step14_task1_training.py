from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import torch
from torch import nn
import segmentation_models_pytorch as smp


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Task1ModelConfig:
    """
    Store all configuration values required to build the Task 1 model.

    The baseline uses a U-Net decoder and an ImageNet-pretrained
    ResNet34 encoder.
    """

    architecture: str = "unet"
    encoder_name: str = "resnet34"
    encoder_weights: str | None = "imagenet"

    in_channels: int = 3
    classes: int = 1

    # Keep activation as None so the model returns raw logits.
    activation: str | None = None


SUPPORTED_ARCHITECTURES = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "fpn": smp.FPN,
}


# ---------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------

def validate_model_config(
    config: Task1ModelConfig,
) -> None:
    """Validate the model configuration before model construction."""
    architecture = config.architecture.lower()

    if architecture not in SUPPORTED_ARCHITECTURES:
        supported_names = ", ".join(
            sorted(SUPPORTED_ARCHITECTURES)
        )

        raise ValueError(
            f"Unsupported architecture: {config.architecture}. "
            f"Supported architectures: {supported_names}"
        )

    if config.in_channels <= 0:
        raise ValueError(
            "in_channels must be greater than zero."
        )

    if config.classes <= 0:
        raise ValueError(
            "classes must be greater than zero."
        )

    if config.classes != 1:
        print(
            "Warning: Task 1 is currently defined as binary lesion "
            "segmentation, so classes=1 is recommended."
        )

    if config.activation is not None:
        print(
            "Warning: activation is not None. "
            "For BCEWithLogitsLoss, the model should return raw logits, "
            "so activation=None is recommended."
        )


def validate_input_size(
    height: int,
    width: int,
    encoder_downsampling_factor: int = 32,
) -> None:
    """
    Confirm that the input size is compatible with encoder downsampling.

    ResNet encoders normally downsample spatial dimensions by a factor
    of 32. Using dimensions divisible by 32 avoids skip-connection
    alignment problems in the decoder.
    """
    if height <= 0 or width <= 0:
        raise ValueError(
            "Input height and width must be greater than zero."
        )

    if (
        height % encoder_downsampling_factor != 0
        or width % encoder_downsampling_factor != 0
    ):
        raise ValueError(
            f"Input size ({height}, {width}) is not divisible by "
            f"{encoder_downsampling_factor}. "
            "Use a size such as 256, 384, or 512."
        )


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------

def build_task1_model(
    config: Task1ModelConfig | None = None,
) -> nn.Module:
    """
    Build the Task 1 binary lesion-segmentation model.

    Parameters
    ----------
    config:
        Model configuration. When omitted, the default baseline is used.

    Returns
    -------
    nn.Module
        A segmentation model returning raw logits with shape
        [batch_size, 1, height, width].
    """
    if config is None:
        config = Task1ModelConfig()

    validate_model_config(
        config
    )

    architecture_name = (
        config.architecture.lower()
    )

    model_class = SUPPORTED_ARCHITECTURES[
        architecture_name
    ]

    model = model_class(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=config.in_channels,
        classes=config.classes,
        activation=config.activation,
    )

    return model


# ---------------------------------------------------------------------
# Encoder control
# ---------------------------------------------------------------------

def set_encoder_trainable(
    model: nn.Module,
    trainable: bool,
) -> None:
    """
    Enable or disable gradient updates for the encoder.

    Freezing the encoder can be useful during an optional warm-up stage.
    The baseline training procedure may also train the full model from
    the beginning.
    """
    if not hasattr(model, "encoder"):
        raise AttributeError(
            "The model does not expose an 'encoder' attribute."
        )

    for parameter in model.encoder.parameters():
        parameter.requires_grad = trainable


def freeze_encoder(
    model: nn.Module,
) -> None:
    """Freeze all encoder parameters."""
    set_encoder_trainable(
        model,
        trainable=False,
    )


def unfreeze_encoder(
    model: nn.Module,
) -> None:
    """Unfreeze all encoder parameters."""
    set_encoder_trainable(
        model,
        trainable=True,
    )


# ---------------------------------------------------------------------
# Parameter inspection
# ---------------------------------------------------------------------

def count_parameters(
    model: nn.Module,
) -> dict[str, int]:
    """Count total and trainable model parameters."""
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameters = (
        total_parameters
        - trainable_parameters
    )

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": frozen_parameters,
    }


def count_encoder_parameters(
    model: nn.Module,
) -> dict[str, int]:
    """Count total and trainable encoder parameters."""
    if not hasattr(model, "encoder"):
        raise AttributeError(
            "The model does not expose an 'encoder' attribute."
        )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.encoder.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    )

    return {
        "encoder_total_parameters": total_parameters,
        "encoder_trainable_parameters": trainable_parameters,
    }


# ---------------------------------------------------------------------
# Forward-pass validation
# ---------------------------------------------------------------------

def get_model_device(
    model: nn.Module,
) -> torch.device:
    """Return the device containing the model parameters."""
    try:
        return next(
            model.parameters()
        ).device

    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def check_forward_pass(
    model: nn.Module,
    input_height: int = 384,
    input_width: int = 384,
    batch_size: int = 1,
) -> dict[str, object]:
    """
    Run a forward-pass test and validate the output shape.

    This test uses random data and does not require the actual dataset.
    """
    validate_input_size(
        input_height,
        input_width,
    )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    device = get_model_device(
        model
    )

    original_training_state = (
        model.training
    )

    model.eval()

    test_input = torch.randn(
        batch_size,
        3,
        input_height,
        input_width,
        device=device,
        dtype=torch.float32,
    )

    logits = model(
        test_input
    )

    expected_shape = (
        batch_size,
        1,
        input_height,
        input_width,
    )

    actual_shape = tuple(
        logits.shape
    )

    if actual_shape != expected_shape:
        raise RuntimeError(
            "Unexpected model output shape.\n"
            f"Expected: {expected_shape}\n"
            f"Actual:   {actual_shape}"
        )

    if not torch.isfinite(
        logits
    ).all():
        raise RuntimeError(
            "The model output contains NaN or infinite values."
        )

    probabilities = torch.sigmoid(
        logits
    )

    result = {
        "input_shape": tuple(
            test_input.shape
        ),
        "output_shape": actual_shape,
        "output_dtype": str(
            logits.dtype
        ),
        "device": str(
            device
        ),
        "minimum_logit": float(
            logits.min().item()
        ),
        "maximum_logit": float(
            logits.max().item()
        ),
        "minimum_probability": float(
            probabilities.min().item()
        ),
        "maximum_probability": float(
            probabilities.max().item()
        ),
    }

    model.train(
        original_training_state
    )

    return result


# ---------------------------------------------------------------------
# Backward-pass validation
# ---------------------------------------------------------------------

def check_backward_pass(
    model: nn.Module,
    input_height: int = 128,
    input_width: int = 128,
    batch_size: int = 1,
) -> dict[str, object]:
    """
    Check whether gradients can pass through the complete model.

    This is only a compatibility test. It is not the final training loop.
    A smaller input size is used to reduce memory usage.
    """
    validate_input_size(
        input_height,
        input_width,
    )

    device = get_model_device(
        model
    )

    original_training_state = (
        model.training
    )

    model.train()
    model.zero_grad(
        set_to_none=True
    )

    test_images = torch.randn(
        batch_size,
        3,
        input_height,
        input_width,
        device=device,
        dtype=torch.float32,
    )

    test_masks = torch.randint(
        low=0,
        high=2,
        size=(
            batch_size,
            1,
            input_height,
            input_width,
        ),
        device=device,
    ).float()

    logits = model(
        test_images
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    loss = criterion(
        logits,
        test_masks,
    )

    if not torch.isfinite(loss):
        raise RuntimeError(
            "The backward-pass test produced a non-finite loss."
        )

    loss.backward()

    parameters_with_gradients = 0
    non_finite_gradient_tensors = 0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        parameters_with_gradients += 1

        if not torch.isfinite(
            parameter.grad
        ).all():
            non_finite_gradient_tensors += 1

    if parameters_with_gradients == 0:
        raise RuntimeError(
            "No model parameter received gradients."
        )

    if non_finite_gradient_tensors > 0:
        raise RuntimeError(
            "Some model gradients contain NaN or infinite values."
        )

    result = {
        "loss": float(
            loss.item()
        ),
        "parameters_with_gradients": (
            parameters_with_gradients
        ),
        "non_finite_gradient_tensors": (
            non_finite_gradient_tensors
        ),
    }

    model.zero_grad(
        set_to_none=True
    )

    model.train(
        original_training_state
    )

    return result


# ---------------------------------------------------------------------
# Configuration saving
# ---------------------------------------------------------------------

def save_model_config(
    config: Task1ModelConfig,
    output_path: Path,
) -> Path:
    """Save the model configuration as a JSON file."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            asdict(config),
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_path


# ---------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------

def main() -> None:
    """
    Build the baseline model and run independent compatibility tests.

    This function does not load the real dataset and does not train
    the model on medical images.
    """
    config = Task1ModelConfig(
        architecture="unet",
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )

    print("=" * 70)
    print("TASK 1 MODEL CONSTRUCTION")
    print("=" * 70)

    print("\nModel configuration:")

    for key, value in asdict(
        config
    ).items():
        print(
            f"  {key}: {value}"
        )

    print("\nBuilding model...")

    model = build_task1_model(
        config
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(
        device
    )

    print(
        f"Model device: {device}"
    )

    parameter_counts = count_parameters(
        model
    )

    encoder_counts = count_encoder_parameters(
        model
    )

    print("\nParameter counts:")

    for key, value in {
        **parameter_counts,
        **encoder_counts,
    }.items():
        print(
            f"  {key}: {value:,}"
        )

    print(
        "\nRunning forward-pass test..."
    )

    forward_result = check_forward_pass(
        model,
        input_height=384,
        input_width=384,
        batch_size=1,
    )

    for key, value in forward_result.items():
        print(
            f"  {key}: {value}"
        )

    print(
        "\nRunning backward-pass test..."
    )

    backward_result = check_backward_pass(
        model,
        input_height=128,
        input_width=128,
        batch_size=1,
    )

    for key, value in backward_result.items():
        print(
            f"  {key}: {value}"
        )

    print("\nEncoder freezing test...")

    freeze_encoder(
        model
    )

    frozen_counts = count_encoder_parameters(
        model
    )

    print(
        "  Trainable encoder parameters after freezing: "
        f"{frozen_counts['encoder_trainable_parameters']:,}"
    )

    unfreeze_encoder(
        model
    )

    unfrozen_counts = count_encoder_parameters(
        model
    )

    print(
        "  Trainable encoder parameters after unfreezing: "
        f"{unfrozen_counts['encoder_trainable_parameters']:,}"
    )

    print("\nAll model checks passed.")


if __name__ == "__main__":
    main()