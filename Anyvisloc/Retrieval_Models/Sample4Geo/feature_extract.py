import torch
import torch.nn.functional as F


def _unwrap_features(features):
    """
    Convert common model outputs into one feature Tensor.
    """

    if isinstance(features, dict):
        for key in (
            "features",
            "embedding",
            "embeddings",
            "global_descriptor",
            "image_features",
        ):
            if key in features:
                features = features[key]
                break
        else:
            tensor_values = [
                value
                for value in features.values()
                if torch.is_tensor(value)
            ]
            if not tensor_values:
                raise TypeError(
                    "Cannot find a Tensor feature in model output dict."
                )
            features = tensor_values[0]

    elif isinstance(features, (tuple, list)):
        tensor_values = [
            value
            for value in features
            if torch.is_tensor(value)
        ]
        if not tensor_values:
            raise TypeError(
                "Cannot find a Tensor feature in model output tuple/list."
            )
        features = tensor_values[0]

    if not torch.is_tensor(features):
        raise TypeError(
            f"Expected Tensor feature, got {type(features)}"
        )

    if features.ndim > 2:
        features = features.flatten(start_dim=1)

    return features


def extract_sample4geo_features(model, images, device):
    images = images.to(device, non_blocking=True)

    with torch.no_grad():
        outputs = model(images)

    features = _unwrap_features(outputs)
    features = F.normalize(features, dim=-1)

    return features