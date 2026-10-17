import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torch.cuda.amp import autocast


def _apply_val_transform(img, val_transforms):
    """
    兼容两种常见的验证变换形式：

    1. torchvision / PIL 风格：
       val_transforms(img)

    2. albumentations 风格：
       val_transforms(image=img)["image"]
    """
    try:
        # 适用于 torchvision.transforms
        return val_transforms(img)
    except Exception:
        # 适用于 albumentations
        transformed = val_transforms(image=np.asarray(img))
        return transformed["image"]


def feature_fusion_all(method_dict, img_feature):
    """
    根据 retrieval_method 提取最终用于检索的特征。

    DAC 与 CAMP 保持一致，使用模型输出的倒数第二项：
        img_feature[-2]
    """
    retrieval_method = method_dict["retrieval_method"]

    if retrieval_method == "MCCG":
        if len(img_feature.shape) == 3:
            fnorm = (
                torch.norm(img_feature, p=2, dim=1, keepdim=True)
                * np.sqrt(img_feature.size(-1))
            )
            img_feature = img_feature.div(fnorm.expand_as(img_feature))
            img_feature = img_feature.view(img_feature.size(0), -1)
        else:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True)
            img_feature = img_feature.div(fnorm.expand_as(img_feature))

    elif retrieval_method == "LPN":
        block = 6
        fnorm = (
            torch.norm(img_feature, p=2, dim=1, keepdim=True)
            * np.sqrt(block)
        )
        img_feature = img_feature.div(fnorm.expand_as(img_feature))
        img_feature = img_feature.view(img_feature.size(0), -1)

    elif retrieval_method == "MIFT":
        if len(img_feature.shape) == 3:
            fnorm = (
                torch.norm(img_feature, p=2, dim=1, keepdim=True)
                * np.sqrt(img_feature.size(-1))
            )
            img_feature = img_feature.div(fnorm.expand_as(img_feature))
            img_feature = img_feature.view(img_feature.size(0), -1)
        else:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True)
            img_feature = img_feature.div(fnorm.expand_as(img_feature))

    elif retrieval_method == "FSRA":
        if len(img_feature.shape) == 3:
            fnorm = (
                torch.norm(img_feature, p=2, dim=1, keepdim=True)
                * np.sqrt(img_feature.size(-1))
            )
            img_feature = img_feature.div(fnorm.expand_as(img_feature))
            img_feature = img_feature.view(img_feature.size(0), -1)
        else:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True)
            img_feature = img_feature.div(fnorm.expand_as(img_feature))

    elif retrieval_method == "CAMP":
        # CAMP 原有接口
        img_feature = img_feature[-2]

    elif retrieval_method == "DAC":
        # DAC 与 CAMP 保持相同的特征提取接口
        # DAC handcrafted ConvNeXt 的检索特征位于模型输出的倒数第二项
        img_feature = img_feature[-2]

    else:
        raise ValueError(
            f"Unsupported retrieval method: {retrieval_method}"
        )

    return img_feature


def extract_features(
    retrieval_size,
    config,
    method_dict,
    MID,
    block_size,
    SATELLITE,
    UAV,
    val_transforms,
    model,
):
    """
    使用 DAC 模型提取卫星图像和 UAV 图像特征。

    参数
    ----
    retrieval_size : int
        输入图像缩放尺寸，例如 384。

    config : dict
        配置字典，至少包含：
            BATCH_SIZE
            DEVICE
            RETRIEVAL_FEATURE_NORM

    method_dict : dict
        方法配置，至少包含：
            retrieval_method = "DAC"

    MID : numpy.ndarray
        卫星图像裁剪中心坐标，通常形状为 [N, 2]。

    block_size : tuple/list
        卫星图像裁剪尺寸，例如：
            (256, 256)

    SATELLITE : numpy.ndarray
        完整卫星图像。

    UAV : torch.Tensor
        一张 UAV 图像，形状通常为：
            [3, H, W]

    val_transforms :
        验证集图像预处理。

    model :
        已加载 DAC 权重的模型。

    返回
    ----
    img_features : torch.Tensor
        所有卫星候选区域的特征，形状：
            [N, D]

    UAV_feature : torch.Tensor
        UAV 查询图像的特征，形状：
            [1, D]
    """

    model.eval()

    device = config["DEVICE"]
    batch_size = config["BATCH_SIZE"]
    retrieval_method = method_dict["retrieval_method"]

    if retrieval_method != "DAC":
        raise ValueError(
            f"This function is intended for DAC, "
            f"but got retrieval_method={retrieval_method}"
        )

    # MID 应为 [N, 2]
    mids = np.asarray(MID).reshape(-1, 2)

    img_features_list = []

    # 分批处理卫星候选区域
    for batch_start in tqdm(
        range(0, len(mids), batch_size),
        desc="Image Retrieval",
    ):
        batch_end = min(batch_start + batch_size, len(mids))
        batch_images = []

        for item in range(batch_start, batch_end):
            mid_x, mid_y = mids[item]

            x_start = int(mid_x - block_size[0] / 2)
            x_end = int(mid_x + block_size[0] / 2)
            y_start = int(mid_y - block_size[1] / 2)
            y_end = int(mid_y + block_size[1] / 2)

            # 防止裁剪坐标超出卫星图像边界
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(SATELLITE.shape[0], x_end)
            y_end = min(SATELLITE.shape[1], y_end)

            satellite_crop = SATELLITE[
                x_start:x_end,
                y_start:y_end,
            ]

            if satellite_crop.size == 0:
                raise RuntimeError(
                    f"Empty satellite crop: "
                    f"({x_start}, {x_end}, {y_start}, {y_end})"
                )

            satellite_crop = cv2.resize(
                satellite_crop,
                (retrieval_size, retrieval_size),
                interpolation=cv2.INTER_LINEAR,
            )

            # 与 CAMP 代码保持一致，先转为 PIL 图像
            satellite_crop = Image.fromarray(satellite_crop)

            # 兼容 torchvision 或 albumentations
            satellite_crop = _apply_val_transform(
                satellite_crop,
                val_transforms,
            )

            # 确保转换结果是 Tensor
            if not torch.is_tensor(satellite_crop):
                satellite_crop = torch.from_numpy(
                    np.asarray(satellite_crop)
                )

            batch_images.append(satellite_crop)

        if len(batch_images) == 0:
            continue

        batch_images = torch.stack(batch_images, dim=0)
        batch_images = batch_images.to(device, non_blocking=True)

        with torch.no_grad():
            # 根据设备决定是否使用 CUDA autocast
            use_amp = str(device).startswith("cuda")

            with autocast(
                enabled=use_amp,
                dtype=torch.float16,
            ):
                # DAC 模型输出通常是 tuple/list
                batch_output = model(batch_images)

                # DAC 特征接口：取输出倒数第二项
                batch_features = feature_fusion_all(
                    method_dict,
                    batch_output,
                )

                if config["RETRIEVAL_FEATURE_NORM"]:
                    batch_features = F.normalize(
                        batch_features,
                        p=2,
                        dim=-1,
                    )

            img_features_list.append(
                batch_features.float()
            )

    # 拼接所有卫星图像特征
    if len(img_features_list) > 0:
        img_features = torch.cat(
            img_features_list,
            dim=0,
        )
    else:
        img_features = torch.empty(
            (0, 0),
            dtype=torch.float32,
            device=device,
        )

    # 提取 UAV 查询图像特征
    with torch.no_grad():
        UAV1 = UAV.to(
            device,
            non_blocking=True,
        )

        # 如果 UAV 是 [C, H, W]，增加 batch 维度
        if UAV1.dim() == 3:
            UAV1 = UAV1.unsqueeze(0)

        elif UAV1.dim() != 4:
            raise ValueError(
                f"UAV tensor must have shape [C,H,W] or [B,C,H,W], "
                f"but got {tuple(UAV1.shape)}"
            )

        use_amp = str(device).startswith("cuda")

        with autocast(
            enabled=use_amp,
            dtype=torch.float16,
        ):
            UAV_output = model(UAV1)

            # DAC 特征接口：取输出倒数第二项
            UAV_feature = feature_fusion_all(
                method_dict,
                UAV_output,
            )

            if config["RETRIEVAL_FEATURE_NORM"]:
                UAV_feature = F.normalize(
                    UAV_feature,
                    p=2,
                    dim=-1,
                )

        UAV_feature = UAV_feature.float()

    return img_features, UAV_feature