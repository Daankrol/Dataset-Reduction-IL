"""EfficientNet in PyTorch.

Reference
    EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def swish(x):
    return x * x.sigmoid()


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class SE(nn.Module):
    """Squeeze-and-Excitation block with Swish."""

    def __init__(self, in_channels, se_channels):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_channels, se_channels, kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = swish(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block(nn.Module):
    """expansion + depthwise + pointwise + squeeze-excitation"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        expand_ratio=1,
        se_ratio=0.0,
        drop_rate=0.0,
    ):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        # Expansion
        channels = expand_ratio * in_channels
        self.conv1 = nn.Conv2d(
            in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        # Depthwise conv
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(1 if kernel_size == 3 else 2),
            groups=channels,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

        # SE layers
        se_channels = int(in_channels * se_ratio)
        self.se = SE(channels, se_channels)

        # Output
        self.conv3 = nn.Conv2d(
            channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection if in and out shapes are the same (MV-V2 style)
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
        out = swish(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.has_skip:
            if self.training and self.drop_rate > 0:
                out = drop_connect(out, self.drop_rate)
            out = out + x
        return out


class EfficientNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.embDim = cfg["out_channels"][-1]
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)
        self.linear = nn.Linear(cfg["out_channels"][-1], num_classes)

    def _make_layers(self, in_channels):
        layers = []
        cfg = [
            self.cfg[k]
            for k in [
                "expansion",
                "out_channels",
                "num_blocks",
                "kernel_size",
                "stride",
            ]
        ]
        b = 0
        blocks = sum(self.cfg["num_blocks"])
        for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg["drop_connect_rate"] * b / blocks
                layers.append(
                    Block(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        expansion,
                        se_ratio=0.25,
                        drop_rate=drop_rate,
                    )
                )
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                out = swish(self.bn1(self.conv1(x)))
                out = self.layers(out)
                out = F.adaptive_avg_pool2d(out, 1)
                e = out.view(out.size(0), -1)
                dropout_rate = self.cfg["dropout_rate"]
                if self.training and dropout_rate > 0:
                    e = F.dropout(e, p=dropout_rate)
        else:
            out = swish(self.bn1(self.conv1(x)))
            out = self.layers(out)
            out = F.adaptive_avg_pool2d(out, 1)
            e = out.view(out.size(0), -1)
            dropout_rate = self.cfg["dropout_rate"]
            if self.training and dropout_rate > 0:
                e = F.dropout(e, p=dropout_rate)
        out = self.linear(e)
        if last:
            return out, e
        else:
            return out

    def get_embedding_dim(self):
        return self.embDim


def EfficientNetB0(num_classes=10):
    cfg = {
        "num_blocks": [1, 2, 2, 3, 3, 4, 1],
        "expansion": [1, 6, 6, 6, 6, 6, 6],
        "out_channels": [16, 24, 40, 80, 112, 192, 320],
        "kernel_size": [3, 3, 5, 3, 5, 5, 3],
        "stride": [1, 2, 2, 2, 1, 2, 1],
        "dropout_rate": 0.2,
        "drop_connect_rate": 0.2,
    }
    return EfficientNet(cfg, num_classes)


# def EfficientNetB0_PyTorch(num_classes=10, pretrained=True, fine_tune=True):
#     # load pretrained weights from the torchvision model
#     model = models.efficientnet_b0(pretrained=pretrained)
#     if fine_tune:
#         for param in model.parameters():
#             param.requires_grad = True
#     elif not fine_tune:
#         for param in model.parameters():
#             param.requires_grad = False
#     # replace the last layer with a new one based on the number of classes
#     model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
#     # make a function on the model that returns the embedding dim
#     model.get_embedding_dim = lambda: 1280
#     model.embDim = 1280

#     # print percentage of trainable parameters depending on the fine_tune flag
#     print(
#         "EfficientNetB0_PyTorch: {}% of {} trainable parameters".format(
#             int(
#                 sum(p.numel() for p in model.parameters() if p.requires_grad)
#                 * 100
#                 / sum(p.numel() for p in model.parameters())
#             ),
#             sum(p.numel() for p in model.parameters()),
#         )
#     )

#     # Redefine the forward() method to def forward(self, x, last=False, freeze=False):
#     model.forward = types.MethodType(forward, model)

#     return model


class EfficientNetB0_PyTorch(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, fine_tune=True):
        super(EfficientNetB0_PyTorch, self).__init__()
        # load pretrained weights from the torchvision model
        self.model = models.efficientnet_b0(pretrained=pretrained)
        self.embedding_recorder = EmbeddingRecorder()
        if fine_tune:
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = False
        # replace the last layer with a new one based on the number of classes
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
        self.embDim = 1280

        # print percentage of trainable parameters depending on the fine_tune flag
        print(
            "EfficientNetB0_PyTorch: {}% of {} trainable parameters".format(
                int(
                    sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    * 100
                    / sum(p.numel() for p in self.model.parameters())
                ),
                sum(p.numel() for p in self.model.parameters()),
            )
        )

    def get_embedding_dim(self):
        return self.embDim

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                out = self.model.features(x)
                out = F.adaptive_avg_pool2d(out, 1)
                e = out.view(out.size(0), -1)
                self.embedding_recorder(e)
                dropout_rate = self.model.classifier[0].p
                if self.training and dropout_rate > 0:
                    e = F.dropout(e, p=dropout_rate)
        else:
            out = self.model.features(x)
            out = F.adaptive_avg_pool2d(out, 1)
            e = out.view(out.size(0), -1)
            self.embedding_recorder(e)
            dropout_rate = self.model.classifier[0].p
            if self.training and dropout_rate > 0:
                e = F.dropout(e, p=dropout_rate)
        out = self.model.classifier(e)
        if last:
            return out, e
        else:
            return out

class EmbeddingRecorder(torch.nn.Module):
    def __init__(self, record_embedding: bool = False):
        super().__init__()
        self.record_embedding = record_embedding

    def forward(self, x):
        if self.record_embedding:
            self.embedding = x
        return x

    def __enter__(self):
        self.record_embedding = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.record_embedding = False


def test():
    # net = EfficientNetB0()
    # x = torch.randn(2, 3, 32, 32)
    # y = net(x)
    # print(y.shape)
    # x = torch.randn

    net = EfficientNetB0_PyTorch(num_classes=10, pretrained=True, fine_tune=True)
    print(net.get_embedding_dim())
    x = torch.randn(2, 3, 33, 33)
    # x.to('cuda')
    y = net(x)
    print(y.shape)
    print(y)


if __name__ == "__main__":
    test()
