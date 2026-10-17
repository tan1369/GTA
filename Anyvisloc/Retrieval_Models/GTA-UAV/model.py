import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class DesModel(nn.Module):
    def __init__(
        self,
        model_name="convnext_base.fb_in22k_ft_in1k_384",
        pretrained=False,
        img_size=384,
        share_weights=True,
        train_with_offset=False,
    ):
        super().__init__()

        self.share_weights = share_weights
        self.model_name = model_name
        self.img_size = int(img_size)

        extra = {}
        if "vit" in model_name.lower() or "swin" in model_name.lower():
            extra["img_size"] = self.img_size

        if share_weights:
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                **extra,
            )
        else:
            self.model1 = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                **extra,
            )
            self.model2 = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                **extra,
            )

        if train_with_offset:
            self.MLP = MLP()

        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / 0.07)
        )

    def load_local_weights(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model", "net"):
                if isinstance(checkpoint.get(key), dict):
                    checkpoint = checkpoint[key]
                    break

        cleaned = {}
        for key, value in checkpoint.items():
            if key.startswith("module."):
                key = key[len("module."):]

            if key.startswith("model."):
                key = key[len("model."):]

            if key.startswith(("head.", "classifier.")):
                continue

            cleaned[key] = value

        target = self.model if self.share_weights else self.model1

        missing, unexpected = target.load_state_dict(
            cleaned,
            strict=False,
        )

        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")

    def get_config(self):
        target = self.model if self.share_weights else self.model1
        return timm.data.resolve_model_data_config(target)

    def set_grad_checkpointing(self, enable=True):
        if self.share_weights:
            if hasattr(self.model, "set_grad_checkpointing"):
                self.model.set_grad_checkpointing(enable)
        else:
            for model in (self.model1, self.model2):
                if hasattr(model, "set_grad_checkpointing"):
                    model.set_grad_checkpointing(enable)

    def freeze_layers(self, *args, **kwargs):
        target = self.model if self.share_weights else self.model1
        for param in target.parameters():
            param.requires_grad = False

    def _encode(self, model, image):
        feature = model(image)

        if isinstance(feature, (tuple, list)):
            feature = feature[0]

        if feature.ndim > 2:
            feature = feature.flatten(start_dim=1)

        return F.normalize(feature.float(), p=2, dim=-1)

    def forward(self, img1=None, img2=None):
        if img1 is None and img2 is None:
            raise ValueError("img1 and img2 cannot both be None")

        if self.share_weights:
            model1 = self.model
            model2 = self.model
        else:
            model1 = self.model1
            model2 = self.model2

        if img1 is not None and img2 is not None:
            return (
                self._encode(model1, img1),
                self._encode(model2, img2),
            )

        if img1 is not None:
            return self._encode(model1, img1)

        return self._encode(model2, img2)

    def offset_pred(self, img_feature1, img_feature2):
        return self.MLP(
            torch.cat((img_feature1, img_feature2), dim=1)
        )


def make_model(
    model_name="vit_base_patch16_rope_reg1_gap_256.sbb_in1k",
    img_size=384,
    pretrained=False,
    local_model_path=None,
):
    model = DesModel(
        model_name=model_name,
        img_size=img_size,
        pretrained=pretrained,
    )

    if local_model_path is not None:
        model.load_local_weights(local_model_path)

    return model