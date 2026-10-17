from pathlib import Path
import torch
from .model import TimmModel

def _extract_state_dict(checkpoint):
    """
    Extract the actual model state_dict from common checkpoint formats.
    """
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
        "net",
        "weights",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    return checkpoint


def _remove_module_prefix(state_dict):
    """
    Remove 'module.' prefix produced by DataParallel/DDP checkpoints.
    """
    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    return cleaned


def get_sample4geo_model(device="cpu"):
    root = Path(__file__).resolve().parent

    checkpoint_path = (
        root
        / "pretrained"
        / "university"
        / "convnext_base.fb_in22k_ft_in1k_384"
        / "weights_e1_0.9515.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Sample4Geo checkpoint not found:\n{checkpoint_path}"
        )

    model = TimmModel(
        "convnext_base.fb_in22k_ft_in1k_384",
        pretrained=False,
        img_size=384,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state_dict = _extract_state_dict(checkpoint)
    state_dict = _remove_module_prefix(state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(
        "[Sample4Geo] checkpoint:",
        checkpoint_path,
    )
    print(
        "[Sample4Geo] missing keys:",
        len(missing_keys),
    )
    print(
        "[Sample4Geo] unexpected keys:",
        len(unexpected_keys),
    )

    if missing_keys:
        print("[Sample4Geo] first missing keys:")
        for key in missing_keys[:10]:
            print("  ", key)

    if unexpected_keys:
        print("[Sample4Geo] first unexpected keys:")
        for key in unexpected_keys[:10]:
            print("  ", key)

    model.to(device)
    model.eval()

    return model