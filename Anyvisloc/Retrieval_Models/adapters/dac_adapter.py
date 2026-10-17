#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision import transforms
from Retrieval_Models.DAC.model import make_model


# ============================================================
# 2. DAC 训练时使用的固定配置
# ============================================================

DEFAULT_IMAGE_SIZE = 384
DEFAULT_BLOCK = 2
DEFAULT_TRIPLET_LOSS = 0.3
DEFAULT_RESNET = False


# ============================================================
# 3. Checkpoint 处理
# ============================================================

def _extract_state_dict(checkpoint):
    """
    兼容多种权重保存格式。
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "DAC checkpoint must be a dictionary, "
            f"but got {type(checkpoint)}"
        )

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
    去除 DataParallel 或 DDP 保存的 module. 前缀。
    """
    result = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]

        result[key] = value

    return result


def _remove_classifier_keys(state_dict):
    """
    删除 GTA-UAV 分类头参数。

    DAC 在 GTA-UAV 上训练时带有分类层，
    但 AnyVisLoc 测试只使用图像检索特征，
    不需要 GTA-UAV 分类器。

    因此只保留主干和特征提取相关权重。
    """
    result = {}

    for key, value in state_dict.items():
        if ".classifier." in key:
            continue

        result[key] = value

    return result


# ============================================================
# 4. 构建并加载 DAC
# ============================================================

def get_DAC_model(device=None):
    """
    AnyVisLoc 的统一模型加载接口。

    注意：这个函数故意不要求传入 config，
    这样不需要修改 avl_utils.py，
    也不会影响 CAMP 的调用方式。

    返回：
        model, val_transforms
    """

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            print(
                "[DAC Adapter] CUDA unavailable, "
                "falling back to CPU."
            )
            device = "cpu"
    default_checkpoint = (
            Path(__file__).resolve().parents[1]
        / "DAC"
            / "hand_convnext"
            / "GTA-UAV"
            / "weights_end.pth"
    )

    checkpoint_path = os.environ.get(
        "DAC_CHECKPOINT",
        str(default_checkpoint),
    )

    checkpoint_path = Path(
        checkpoint_path
    ).expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "DAC checkpoint does not exist:\n"
            f"{checkpoint_path}"
        )

    print("[DAC Adapter] checkpoint:")
    print(f"  {checkpoint_path}")

    print("[DAC Adapter] device:")
    print(f"  {device}")

    # 这些参数必须与 train_gta_uav.py 中的配置一致。
    model_config = SimpleNamespace(
        views=2,
        nclasses=1,
        block=DEFAULT_BLOCK,
        triplet_loss=DEFAULT_TRIPLET_LOSS,
        resnet=DEFAULT_RESNET,
    )

    # 创建外部 DAC 项目中的 handcrafted ConvNeXt。
    model = make_model(model_config)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state_dict = _extract_state_dict(checkpoint)
    state_dict = _remove_module_prefix(state_dict)
    state_dict = _remove_classifier_keys(state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(
    state_dict,
    strict=False,
)

    unexpected_missing = [
        k for k in missing_keys
        if ".classifier." not in k
    ]

    unexpected_extra = [
        k for k in unexpected_keys
        if ".classifier." not in k
    ]

    if unexpected_missing or unexpected_extra:
        raise RuntimeError(
            "DAC non-classifier weights are not fully matched:\n"
            f"missing={unexpected_missing[:20]}\n"
            f"unexpected={unexpected_extra[:20]}"
        )

    print("[DAC Adapter] classifier keys ignored")
    print("[DAC Adapter] non-classifier weights matched")

    print(
        "[DAC Adapter] missing keys:",
        len(missing_keys),
    )

    print(
        "[DAC Adapter] unexpected keys:",
        len(unexpected_keys),
    )

    if missing_keys:
        print(
            "[DAC Adapter] missing keys preview:"
        )
        for key in missing_keys[:10]:
            print(f"  {key}")

    if unexpected_keys:
        print(
            "[DAC Adapter] unexpected keys preview:"
        )
        for key in unexpected_keys[:10]:
            print(f"  {key}")

    model = model.to(device)
    model.eval()

    # DAC 训练和测试采用 ImageNet 归一化。
    val_transforms = transforms.Compose(
        [
            transforms.Resize(
                (
                    DEFAULT_IMAGE_SIZE,
                    DEFAULT_IMAGE_SIZE,
                ),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    return model, val_transforms


class DACAdapter:
    """AnyVisLoc adapter exposing raw-image and preprocessed-batch encoding."""

    name = "DAC"

    def __init__(self, device="cuda", image_size=384):
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            print("[DAC] CUDA unavailable, using CPU.")
            device = "cpu"
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.model, self.transform = get_DAC_model(self.device)

    def preprocess(self, image):
        from PIL import Image
        import numpy as np
        if isinstance(image, Image.Image):
            return self.transform(image.convert("RGB"))
        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
                raise ValueError(f"numpy image must be [H,W,C], got {arr.shape}")
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            elif arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255 if arr.max() <= 1 else arr, 0, 255).astype(np.uint8)
            return self.transform(Image.fromarray(arr))
        if isinstance(image, torch.Tensor):
            if image.ndim != 3:
                raise ValueError(f"Tensor image must be [C,H,W], got {tuple(image.shape)}")
            tensor = image.detach().cpu().float()
            if tensor.shape[0] not in (1, 3, 4) and tensor.shape[-1] in (1, 3, 4):
                tensor = tensor.permute(2, 0, 1)
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            elif tensor.shape[0] == 4:
                tensor = tensor[:3]
            elif tensor.shape[0] != 3:
                raise ValueError(f"Tensor image must have 1, 3, or 4 channels, got {tuple(tensor.shape)}")
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=(self.image_size, self.image_size),
                mode="bicubic", align_corners=False,
            ).squeeze(0)
            mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = tensor.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return (tensor - mean) / std
        raise TypeError(f"Unsupported image type: {type(image)}")

    @torch.inference_mode()
    def encode_batch(self, images):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"DAC batch must be [B,3,H,W], got {tuple(images.shape)}")
        inputs = images.to(self.device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            outputs = self.model(inputs)
            
        # DAC eval output is (global GAP, part feature map); use global GAP.
        features = outputs[-2] if isinstance(outputs, (tuple, list)) else outputs
        if not hasattr(self, "_feature_debug_printed"):
            print("[DAC] output type:", type(outputs))
            print("[DAC] gap feature shape:", tuple(features.shape))
            print("[DAC] feature finite:", bool(torch.isfinite(features).all()))
            print("[DAC] feature norm:",
                features.float().norm(dim=-1).mean().item())
            self._feature_debug_printed = True
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        return torch.nn.functional.normalize(features.float(), dim=-1)

    @torch.inference_mode()
    def encode(self, images):
        if isinstance(images, (list, tuple)):
            batch = torch.stack([self.preprocess(x) for x in images])
        elif isinstance(images, torch.Tensor) and images.ndim == 4:
            batch = torch.stack([self.preprocess(x) for x in images])
        else:
            batch = self.preprocess(images).unsqueeze(0)
        return self.encode_batch(batch)
