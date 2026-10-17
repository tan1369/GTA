from pathlib import Path
import importlib.util

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class WaveAdapter:
    name = "Wave"

    def __init__(
        self,
        device="cuda",
        image_size=384,
        weight_type="cross",
        model_name="vit_base_patch16_rope_reg1_gap_256",
    ):
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            print("[Wave] CUDA unavailable, using CPU.")
            device = "cpu"

        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.weight_type = str(weight_type).lower()

        if self.weight_type not in ("cross", "same"):
            raise ValueError(
                "weight_type must be 'cross' or 'same'"
            )

        retrieval_root = Path(__file__).resolve().parents[1]
        wave_root = retrieval_root / "wave-first-work"
        model_path = wave_root / "model.py"

        weight_path = (
            wave_root
            / "pretrained"
            / self.weight_type
            / f"{self.weight_type}_checkpoint.pth"
        )

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Wave model.py not found: {model_path}"
            )

        if not weight_path.is_file():
            raise FileNotFoundError(
                f"Wave checkpoint not found: {weight_path}"
            )

        wave_module = self._load_model_module(model_path)

        self.model = wave_module.make_model(
            model_name=model_name,
            img_size=self.image_size,
            pretrained=False,
            local_model_path=str(weight_path),
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        self.image_size,
                        self.image_size,
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

        print(f"[Wave] model: {model_name}")
        print(f"[Wave] weight type: {self.weight_type}")
        print(f"[Wave] checkpoint: {weight_path}")
        print(f"[Wave] device: {self.device}")

    @staticmethod
    def _load_model_module(model_path):
        spec = importlib.util.spec_from_file_location(
            "wave_first_work_model",
            str(model_path),
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                f"Cannot import Wave model: {model_path}"
            )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def preprocess(self, image):
        """
        Input must be RGB.

        Supports:
            PIL.Image
            numpy.ndarray [H, W, 3]
            torch.Tensor [3, H, W]
        """

        if isinstance(image, Image.Image):
            return self.transform(image.convert("RGB"))

        if isinstance(image, np.ndarray):
            array = image

            if array.ndim == 2:
                array = np.repeat(
                    array[..., None],
                    3,
                    axis=2,
                )

            if array.ndim != 3:
                raise ValueError(
                    f"Expected numpy image [H,W,C], got {array.shape}"
                )

            if array.shape[2] == 1:
                array = np.repeat(
                    array,
                    3,
                    axis=2,
                )
            elif array.shape[2] == 4:
                array = array[:, :, :3]
            elif array.shape[2] != 3:
                raise ValueError(
                    f"Expected 1, 3, or 4 channels, got {array.shape}"
                )

            if array.dtype != np.uint8:
                array = array.astype(np.float32)

                if array.max() <= 1.0:
                    array = array * 255.0

                array = np.clip(
                    array,
                    0.0,
                    255.0,
                ).astype(np.uint8)

            return self.transform(Image.fromarray(array))

        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu().float()

            if tensor.ndim != 3:
                raise ValueError(
                    f"Expected tensor [C,H,W], got {tuple(tensor.shape)}"
                )

            if tensor.shape[0] not in (1, 3, 4):
                if tensor.shape[-1] in (1, 3, 4):
                    tensor = tensor.permute(2, 0, 1)
                else:
                    raise ValueError(
                        f"Invalid tensor shape: {tuple(tensor.shape)}"
                    )

            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            elif tensor.shape[0] == 4:
                tensor = tensor[:3]

            if tensor.max() > 1.0:
                tensor = tensor / 255.0

            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(
                    self.image_size,
                    self.image_size,
                ),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)

            mean = tensor.new_tensor(
                [
                    0.485,
                    0.456,
                    0.406,
                ]
            ).view(3, 1, 1)

            std = tensor.new_tensor(
                [
                    0.229,
                    0.224,
                    0.225,
                ]
            ).view(3, 1, 1)

            return (tensor - mean) / std

        raise TypeError(
            f"Unsupported image type: {type(image)}"
        )

    @torch.inference_mode()
    def encode_batch(self, images):
        """
        Input:
            preprocessed Tensor [B, 3, 384, 384]

        Output:
            normalized Tensor [B, 768]
        """

        if images.ndim == 3:
            images = images.unsqueeze(0)

        if images.ndim != 4:
            raise ValueError(
                f"Expected [B,C,H,W], got {tuple(images.shape)}"
            )

        if images.shape[1] != 3:
            raise ValueError(
                f"Wave expects RGB input, got {tuple(images.shape)}"
            )

        images = images.to(
            self.device,
            non_blocking=True,
        )

        features = self.model(images)

        if isinstance(features, (tuple, list)):
            features = features[0]

        if features.ndim == 1:
            features = features.unsqueeze(0)

        if features.ndim > 2:
            features = features.flatten(start_dim=1)

        features = features.float()

        return F.normalize(
            features,
            p=2,
            dim=-1,
        )

    @torch.inference_mode()
    def encode(self, images):
        """
        Input:
            raw RGB image or raw RGB image list

        Output:
            normalized Tensor [B, 768]
        """

        if isinstance(
            images,
            (
                Image.Image,
                np.ndarray,
            ),
        ):
            images = [images]

        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images = images.unsqueeze(0)

            if images.ndim != 4:
                raise ValueError(
                    f"Expected tensor [B,C,H,W], got {tuple(images.shape)}"
                )

            batch = torch.stack(
                [
                    self.preprocess(image)
                    for image in images
                ],
                dim=0,
            )

            return self.encode_batch(batch)

        if isinstance(images, (list, tuple)):
            if len(images) == 0:
                raise ValueError(
                    "Cannot encode an empty image list."
                )

            batch = torch.stack(
                [
                    self.preprocess(image)
                    for image in images
                ],
                dim=0,
            )

            return self.encode_batch(batch)

        raise TypeError(
            f"Unsupported input type: {type(images)}"
        )