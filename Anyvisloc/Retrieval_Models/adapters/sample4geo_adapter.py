from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from Retrieval_Models.Sample4Geo.loader import get_sample4geo_model
from Retrieval_Models.Sample4Geo.feature_extract import (
    extract_sample4geo_features,
)


class Sample4GeoAdapter:
    """
    Sample4Geo 到 AnyVisLoc 的适配器。

    对外提供两个主要接口：

    1. encode(...)
       输入原始图像，适配器自动完成预处理和特征提取。

    2. encode_batch(...)
       输入已经完成预处理的 Tensor，
       只执行模型推理和特征提取。
    """

    name = "Sample4Geo"

    def __init__(
        self,
        device="cuda",
        image_size=384,
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[Sample4Geo] CUDA unavailable, using CPU.")
            device = "cpu"

        self.device = torch.device(device)
        self.image_size = image_size

        print(f"[Sample4Geo] loading model on {self.device}")

        self.model = get_sample4geo_model(
            device=str(self.device)
        )

        self.model.eval()

        # Sample4Geo 使用 ImageNet 归一化。
        # 输入尺寸为 384 x 384。
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def preprocess(self, image):
        """
        将一张原始图像转换成：

            [3, 384, 384]

        支持：

            PIL.Image
            numpy.ndarray
            torch.Tensor

        注意：
            Tensor 输入必须是未归一化的原始图像，
            数值范围可以是 [0, 1] 或 [0, 255]。
        """

        # 1. PIL.Image
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            return self.transform(image)

        # 2. numpy.ndarray
        if isinstance(image, np.ndarray):
            array = image

            if array.ndim == 2:
                # 灰度图转 RGB
                array = np.repeat(
                    array[..., None],
                    3,
                    axis=2,
                )

            if array.ndim != 3:
                raise ValueError(
                    "numpy image must have shape [H,W,C] "
                    f"or [H,W], got {array.shape}"
                )

            if array.shape[2] == 1:
                array = np.repeat(
                    array,
                    3,
                    axis=2,
                )
            elif array.shape[2] == 4:
                # RGBA 只保留 RGB
                array = array[:, :, :3]
            elif array.shape[2] != 3:
                raise ValueError(
                    "numpy image must have 1, 3, or 4 channels, "
                    f"got {array.shape}"
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

            image = Image.fromarray(array)
            return self.transform(image)

        # 3. torch.Tensor
        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()

            if tensor.ndim != 3:
                raise ValueError(
                    "Tensor image must have shape [C,H,W] "
                    f"or [H,W,C], got {tuple(tensor.shape)}"
                )

            # [H,W,C] 转为 [C,H,W]
            if tensor.shape[-1] in (1, 3, 4):
                tensor = tensor.permute(2, 0, 1)

            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            elif tensor.shape[0] == 4:
                tensor = tensor[:3]
            elif tensor.shape[0] != 3:
                raise ValueError(
                    "Tensor image must have 1, 3, or 4 channels, "
                    f"got {tuple(tensor.shape)}"
                )

            tensor = tensor.float()

            if tensor.max() > 1.0:
                tensor = tensor / 255.0

            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(
                    self.image_size,
                    self.image_size,
                ),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)

            mean = torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=tensor.dtype,
            ).view(3, 1, 1)

            std = torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=tensor.dtype,
            ).view(3, 1, 1)

            tensor = (tensor - mean) / std

            return tensor

        raise TypeError(
            "Unsupported image type: "
            f"{type(image)}. Expected PIL.Image, "
            "numpy.ndarray, or torch.Tensor."
        )

    def preprocess_batch(
        self,
        images: Sequence,
    ):
        """
        对一组原始图像做预处理。

        输入：
            PIL/numpy/Tensor 图像列表

        输出：
            Tensor，形状为 [B,3,384,384]
        """

        tensors = [
            self.preprocess(image)
            for image in images
        ]

        if len(tensors) == 0:
            raise ValueError(
                "Cannot preprocess an empty image batch."
            )

        return torch.stack(
            tensors,
            dim=0,
        )

    @torch.inference_mode()
    def encode_batch(self, images):
        """
        对已经预处理好的 Tensor 做特征提取。

        输入：

            images: [B,3,384,384]

        输出：

            features: [B,D]

        这里的 images 必须已经完成：

            Resize
            ToTensor
            ImageNet Normalize
        """

        if not isinstance(images, torch.Tensor):
            raise TypeError(
                "encode_batch expects torch.Tensor, got "
                f"{type(images)}"
            )

        if images.ndim == 3:
            images = images.unsqueeze(0)

        if images.ndim != 4:
            raise ValueError(
                "encode_batch expects [B,C,H,W], got "
                f"{tuple(images.shape)}"
            )

        if images.shape[1] != 3:
            raise ValueError(
                "Sample4Geo expects 3-channel RGB images, got "
                f"{tuple(images.shape)}"
            )

        images = images.to(
            self.device,
            non_blocking=True,
        )

        # 调用你已有的 Sample4Geo 特征提取文件。
        # 不调用 CAMP 的 feature_extract.py。
        features = extract_sample4geo_features(
            self.model,
            images,
            self.device,
        )

        if not isinstance(features, torch.Tensor):
            raise TypeError(
                "extract_sample4geo_features must return Tensor, got "
                f"{type(features)}"
            )

        if features.ndim == 1:
            features = features.unsqueeze(0)

        if features.ndim != 2:
            raise ValueError(
                "Sample4Geo feature must have shape [B,D], got "
                f"{tuple(features.shape)}"
            )

        return features

    @torch.inference_mode()
    def encode(self, images):
        """
        对原始图像做预处理并提取特征。

        支持：

            单张 PIL/numpy/Tensor
            PIL/numpy/Tensor 图像列表

        输出：

            [B,D]
        """

        # 单张原始图像
        if isinstance(
            images,
            (
                Image.Image,
                np.ndarray,
            ),
        ):
            images = [images]

        # Tensor 可以是单张 [C,H,W] 或批量 [B,C,H,W]
        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images = images.unsqueeze(0)

            if images.ndim != 4:
                raise ValueError(
                    "Tensor batch must have shape [B,C,H,W], got "
                    f"{tuple(images.shape)}"
                )

            # 这里把 Tensor 当作原始图像处理，
            # 因此会执行 Resize 和 Normalize。
            preprocessed = torch.stack(
                [
                    self.preprocess(image)
                    for image in images
                ],
                dim=0,
            )

            return self.encode_batch(preprocessed)

        # 图像列表
        if isinstance(images, (list, tuple)):
            preprocessed = self.preprocess_batch(images)
            return self.encode_batch(preprocessed)

        raise TypeError(
            "Unsupported input type for encode: "
            f"{type(images)}"
        )