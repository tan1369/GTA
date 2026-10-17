import torch.nn as nn


def c2_msra_fill(module: nn.Module) -> None:
    nn.init.kaiming_normal_(
        module.weight,
        mode="fan_out",
        nonlinearity="relu",
    )

    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def normal_init(module: nn.Module, std=0.01):
    nn.init.normal_(module.weight, std=std)

    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def init_weights(module, init_linear="normal"):
    if init_linear not in ("normal", "kaiming"):
        raise ValueError(
            f"Undefined init_linear: {init_linear}"
        )

    for m in module.modules():
        if isinstance(m, nn.Linear):
            if init_linear == "normal":
                normal_init(m, std=0.01)
            else:
                c2_msra_fill(m)

        elif isinstance(
            m,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.GroupNorm,
                nn.SyncBatchNorm,
            ),
        ):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Conv1d):
            c2_msra_fill(m)