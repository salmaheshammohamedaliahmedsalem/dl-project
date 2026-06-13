# model_2d_resunetpp_multihead.py
# Shared encoder + source-specific foreground decoder heads for model-aligned 2D.
# Output: one-vs-rest foreground logits (C-1 channels per source).

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, ch, reduction=16):
        super().__init__()
        hidden = max(ch // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, ch),
            nn.Sigmoid(),
        )
    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ResidualConvBlock(nn.Module):
    """Residual conv block with optional SE."""
    def __init__(self, in_ch, out_ch, stride=1, use_se=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch or stride != 1 else nn.Identity()
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.se(self.conv(x)) + self.skip(x))


class ASPP2D(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
        super().__init__()
        branches = []
        for r in rates:
            if r == 1:
                branches.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)))
            else:
                branches.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)))
        self.branches = nn.ModuleList(branches)
        self.fuse = nn.Sequential(
            nn.Conv2d(len(rates) * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.fuse(torch.cat([b(x) for b in self.branches], dim=1))


class AttentionGate(nn.Module):
    def __init__(self, skip_ch, gate_ch, inter_ch):
        super().__init__()
        self.theta = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.phi = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.psi = nn.Conv2d(inter_ch, 1, 1)

    def forward(self, skip, gate):
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return skip * torch.sigmoid(self.psi(F.relu(self.theta(skip) + self.phi(gate), inplace=True)))


class SharedEncoder2D(nn.Module):
    def __init__(self, in_channels=3, filters=None):
        super().__init__()
        if filters is None:
            filters = [32, 64, 128, 256, 512]
        f = filters
        self.stem = nn.Sequential(ConvBNReLU(in_channels, f[0]), ConvBNReLU(f[0], f[0]))
        self.down1 = ResidualConvBlock(f[0], f[1], stride=2)
        self.down2 = ResidualConvBlock(f[1], f[2], stride=2)
        self.down3 = ResidualConvBlock(f[2], f[3], stride=2)
        self.bridge = ASPP2D(f[3], f[4])

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        b = self.bridge(s4)
        return s1, s2, s3, s4, b


class ForegroundDecoder2D(nn.Module):
    """Decoder producing one-vs-rest foreground logits."""
    def __init__(self, num_fg_classes, filters=None, dropout=0.1):
        super().__init__()
        if filters is None:
            filters = [32, 64, 128, 256, 512]
        f = filters

        self.up3 = nn.ConvTranspose2d(f[4], f[3], 2, 2)
        self.ag3 = AttentionGate(f[3], f[3], f[2])
        self.dec3 = ResidualConvBlock(f[3] * 2, f[3])

        self.up2 = nn.ConvTranspose2d(f[3], f[2], 2, 2)
        self.ag2 = AttentionGate(f[2], f[2], f[1])
        self.dec2 = ResidualConvBlock(f[2] * 2, f[2])

        self.up1 = nn.ConvTranspose2d(f[2], f[1], 2, 2)
        self.ag1 = AttentionGate(f[1], f[1], f[0])
        self.dec1 = ResidualConvBlock(f[1] * 2, f[1])

        self.aspp_out = ASPP2D(f[1], f[0], rates=(1, 3, 6))
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(f[0], num_fg_classes, 1)

    def forward(self, s1, s2, s3, s4, b):
        d3 = self.up3(b)
        if d3.shape[-2:] != s4.shape[-2:]:
            d3 = F.interpolate(d3, size=s4.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, self.ag3(s4, d3)], dim=1))

        d2 = self.up2(d3)
        if d2.shape[-2:] != s3.shape[-2:]:
            d2 = F.interpolate(d2, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, self.ag2(s3, d2)], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != s2.shape[-2:]:
            d1 = F.interpolate(d1, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, self.ag1(s2, d1)], dim=1))

        out = self.aspp_out(d1)
        out = self.drop(out)

        # Upsample to match s1 (full input resolution)
        if out.shape[-2:] != s1.shape[-2:]:
            out = F.interpolate(out, size=s1.shape[-2:], mode="bilinear", align_corners=False)

        return self.head(out)


class MultiHeadResUNetPP2D(nn.Module):
    """Shared encoder + per-source one-vs-rest foreground decoders."""
    def __init__(self, in_channels=3, source_order=None, num_classes_by_source=None, filters=None, dropout=0.1):
        super().__init__()
        if source_order is None:
            source_order = ["2ch", "4ch", "sax"]
        if num_classes_by_source is None:
            num_classes_by_source = {"2ch": 3, "4ch": 6, "sax": 4}
        if filters is None:
            filters = [32, 64, 128, 256, 512]

        self.source_order = source_order
        self.num_classes_by_source = num_classes_by_source
        self.encoder = SharedEncoder2D(in_channels, filters)
        self.decoders = nn.ModuleDict({
            src: ForegroundDecoder2D(
                num_fg_classes=nc - 1,  # one-vs-rest: exclude background
                filters=filters,
                dropout=dropout,
            )
            for src, nc in num_classes_by_source.items()
        })

    def forward(self, x, source):
        if source not in self.decoders:
            raise ValueError(f"Unknown source: {source}. Expected one of {list(self.decoders.keys())}")
        s1, s2, s3, s4, b = self.encoder(x)
        return self.decoders[source](s1, s2, s3, s4, b)


if __name__ == "__main__":
    model = MultiHeadResUNetPP2D()
    x = torch.randn(2, 3, 256, 256)
    for src, nc in {"2ch": 3, "4ch": 6, "sax": 4}.items():
        out = model(x, src)
        print(f"{src}: input {x.shape} -> output {out.shape} (expected {nc-1} fg channels)")
        assert out.shape == (2, nc - 1, 256, 256)
    print("Smoke test passed!")
