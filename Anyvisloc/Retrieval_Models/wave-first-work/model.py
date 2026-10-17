import torch
import torch.nn as nn
import timm


class DesModel(nn.Module):
    """
    Game4Loc retrieval-only descriptor.

    Input:
        [B, 3, H, W], RGB, ImageNet normalized

    Output:
        [B, D], image descriptor
    """

    def __init__(
        self,
        model_name="vit_base_patch16_rope_reg1_gap_256",
        img_size=384,
        pretrained=False,
        local_model_path=None,
    ):
        super().__init__()

        self.model_name = model_name
        self.img_size = int(img_size)

        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            img_size=img_size,
        )

        if local_model_path is not None:
            self.load_local_weights(local_model_path)

        self.embed_dim = self.model.embed_dim

    def load_local_weights(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        if isinstance(checkpoint, dict):
            for key in (
                "state_dict",
                "model_state_dict",
                "model",
                "net",
            ):
                if isinstance(checkpoint.get(key), dict):
                    checkpoint = checkpoint[key]
                    break

        state_dict = {}

        for key, value in checkpoint.items():
            if key.startswith("module."):
                key = key[len("module."):]

            # 如果权重来自完整的 DesModel，去掉 model. 前缀
            if key.startswith("model."):
                key = key[len("model."):]

            # 分类头不参与检索
            if key.startswith("head."):
                continue

            state_dict[key] = value

        missing, unexpected = self.model.load_state_dict(
            state_dict,
            strict=False,
        )

        print(
            f"[Game4Loc] loaded weights: {checkpoint_path}"
        )
        print(
            f"[Game4Loc] missing keys: {len(missing)}"
        )
        print(
            f"[Game4Loc] unexpected keys: {len(unexpected)}"
        )

    def get_patches(self, x):
        """
        Extract ViT patch tokens.

        Returns:
            [B, N, C]
        """
        model = self.model

        x = model.patch_embed(x)

        rope_embed = None

        if hasattr(model, "_pos_embed"):
            pos_out = model._pos_embed(x)

            if isinstance(pos_out, tuple):
                x, rope_embed = pos_out
            else:
                x = pos_out
        else:
            if hasattr(model, "cls_token") and model.cls_token is not None:
                cls_token = model.cls_token.expand(
                    x.shape[0],
                    -1,
                    -1,
                )
                x = torch.cat((cls_token, x), dim=1)

            if hasattr(model, "pos_embed") and model.pos_embed is not None:
                x = x + model.pos_embed

            if hasattr(model, "pos_drop"):
                x = model.pos_drop(x)

        for block in model.blocks:
            if rope_embed is not None:
                x = block(x, rope=rope_embed)
            else:
                x = block(x)

        return x

    def pool_tokens(self, tokens):
        """
        Convert ViT tokens to one global descriptor.
        """
        model = self.model

        if hasattr(model, "norm") and model.norm is not None:
            tokens = model.norm(tokens)

        has_cls_token = (
            getattr(model, "cls_token", None) is not None
        )

        num_prefix = getattr(
            model,
            "num_prefix_tokens",
            1 if has_cls_token else 0,
        )

        has_prefix = (
            num_prefix > 0
            and tokens.shape[1] >= num_prefix
        )

        patch_tokens = (
            tokens[:, num_prefix:]
            if has_prefix
            else tokens
        )

        pool_type = getattr(
            model,
            "global_pool",
            "token",
        )

        if pool_type == "avg":
            feature = patch_tokens.mean(dim=1)
        elif has_cls_token and has_prefix:
            feature = tokens[:, 0]
        else:
            feature = patch_tokens.mean(dim=1)

        if hasattr(model, "fc_norm") and model.fc_norm is not None:
            feature = model.fc_norm(feature)

        if hasattr(model, "pre_logits") and model.pre_logits is not None:
            feature = model.pre_logits(feature)

        return feature

    @torch.inference_mode()
    def forward(self, x):
        """
        x:
            [B, 3, 384, 384]

        return:
            [B, D]
        """
        tokens = self.get_patches(x)
        feature = self.pool_tokens(tokens)

        return torch.nn.functional.normalize(
            feature.float(),
            p=2,
            dim=-1,
        )


def make_model(
    model_name="vit_base_patch16_rope_reg1_gap_256",
    img_size=384,
    pretrained=False,
    local_model_path=None,
):
    return DesModel(
        model_name=model_name,
        img_size=img_size,
        pretrained=pretrained,
        local_model_path=local_model_path,
    )