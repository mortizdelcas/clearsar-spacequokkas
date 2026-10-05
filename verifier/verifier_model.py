import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.rand(shape, dtype=x.dtype, device=x.device).add_(keep).floor_()
    return x.div(keep) * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, mid, 1, bias=True)
        self.fc2 = nn.Conv2d(mid, channels, 1, bias=True)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ResConvBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1, drop_path_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)
        self.se = SEBlock(c_out)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.skip = nn.Identity()
        if c_in != c_out or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.drop_path(out) + identity
        return self.act(out)


class VerifierV2(nn.Module):
    """ResidualSE CNN + dual-pool + MLP head.

    Inputs:
      crop: (B, 3, 96, 96) uint8 (normalized inside the model)
      feat: (B, F) float (z-scored upstream)
    Output:
      logit: (B,) -- sigmoid yields P(box is a true positive at IoU>=0.5).
    """
    def __init__(self, n_feat=22, base=48, drop_path_rate=0.1, head_drop=0.3):
        super().__init__()
        dpr = [drop_path_rate * i / 3 for i in range(4)]

        self.stem = ResConvBlock(3, base, stride=2, drop_path_rate=dpr[0])
        self.b1 = ResConvBlock(base, base * 2, stride=2, drop_path_rate=dpr[1])
        self.b2 = ResConvBlock(base * 2, base * 4, stride=2, drop_path_rate=dpr[2])
        self.b3 = ResConvBlock(base * 4, base * 8, stride=2, drop_path_rate=dpr[3])

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        img_dim = base * 8 * 2

        feat_hidden = 128
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, feat_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feat_hidden, feat_hidden),
            nn.SiLU(inplace=True),
        )
        self.feat_skip = nn.Linear(n_feat, feat_hidden) if n_feat != feat_hidden else nn.Identity()

        head_in = img_dim + feat_hidden
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(head_drop),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(head_drop * 0.5),
            nn.Linear(128, 1),
        )

        self.register_buffer("mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, crop, feat):
        if crop.dtype != torch.float32:
            crop = crop.float() / 255.0
        x = (crop - self.mean) / self.std
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x_avg = self.avg_pool(x).flatten(1)
        x_max = self.max_pool(x).flatten(1)
        img_feat = torch.cat([x_avg, x_max], dim=1)

        f = self.feat_mlp(feat) + self.feat_skip(feat)
        z = torch.cat([img_feat, f], dim=1)
        return self.head(z).squeeze(-1)


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class Verifier(nn.Module):
    """Legacy verifier architecture (kept for backwards compatibility)."""
    def __init__(self, n_feat=22, base=48):
        super().__init__()
        self.stem = ConvBlock(3, base, stride=2)
        self.b1 = ConvBlock(base, base * 2, stride=2)
        self.b2 = ConvBlock(base * 2, base * 4, stride=2)
        self.b3 = ConvBlock(base * 4, base * 8, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 64), nn.SiLU(inplace=True),
            nn.Linear(64, 64), nn.SiLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(base * 8 + 64, 128), nn.SiLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(128, 1),
        )
        self.register_buffer("mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, crop, feat):
        if crop.dtype != torch.float32:
            crop = crop.float() / 255.0
        x = (crop - self.mean) / self.std
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.pool(x).flatten(1)
        f = self.feat_mlp(feat)
        z = torch.cat([x, f], dim=1)
        return self.head(z).squeeze(-1)
