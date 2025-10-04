import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution:
    depthwise 3x3 (groups=in_channels) + pointwise 1x1
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, dropout=0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=1, padding=padding, dilation=dilation,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        return x


class DilatedConv2d(nn.Module):
    """
    Dilated 3x3 conv (padding=dilation to preserve HxW)
    """
    def __init__(self, in_channels, out_channels, dilation=2, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=1, padding=dilation, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        return x


class TransitionBlock(nn.Module):
    """
    Transition: 1x1 conv (channel adjust) -> 3x3 conv stride=2 (downsample)
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.down = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.reduce(x)
        x = self.down(x)
        return x


class CIFAR10Netv2(nn.Module):
    """≈ 195 K params — 4 conv blocks + transitions + GAP"""
    def __init__(self, num_classes=10, dropout=0.1):
        super().__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(), nn.Dropout(dropout),
        )

        # Transition 1  16 → 32
        self.trans1 = TransitionBlock(16, 32, dropout)

        # Block 2  32 → 32
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout(dropout),
        )

        # Transition 2  32 → 52
        self.trans2 = TransitionBlock(32, 52, dropout)

        # Block 3  52 → 52 (dilated + depthwise)
        self.block3 = nn.Sequential(
            nn.Conv2d(52, 52, 3, padding=1, bias=False),
            nn.BatchNorm2d(52), nn.ReLU(), nn.Dropout(dropout),
            DilatedConv2d(52, 52, dilation=2, dropout=dropout),
            DepthwiseSeparableConv2d(52, 52, 3, padding=1, dropout=dropout),
        )

        # Transition 3  52 → 52 (stride 2)
        self.trans3 = TransitionBlock(52, 52, dropout)

        # Block 4  4 convs before GAP
        self.block4 = nn.Sequential(
            nn.Conv2d(52, 52, 3, padding=1, bias=False),
            nn.BatchNorm2d(52), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv2d(52, 52, 1, bias=False),
            nn.BatchNorm2d(52), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv2d(52, 52, 1, bias=False),
            nn.BatchNorm2d(52), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv2d(52, 52, 1, bias=False),
            nn.BatchNorm2d(52), nn.ReLU(), nn.Dropout(dropout),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(52, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d,)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)
    
    def get_torch_summary(self):
        from torchsummary import summary
        # OLD CODE: use_cuda = torch.cuda.is_available()
        # OLD CODE: device = torch.device("cuda" if use_cuda else "cpu")
        # NEW CODE: Check for Apple MPS (Metal Performance Shaders) availability
        use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
        device = torch.device("mps" if use_mps else "cpu")
        model = CIFAR10Netv2().to(device)
        summary(model, input_size=(3, 44, 44))

    def forward(self, x):
        # 32x32
        x = self.block1(x)              # -> 32x32
        x = self.trans1(x)              # -> 16x16
        x = self.block2(x)              # -> 16x16
        x = self.trans2(x)              # -> 8x8
        x = self.block3(x)              # -> 8x8 (dilated + depthwise inside)
        x = self.trans3(x)              # -> 4x4 (stride-2)
        x = self.block4(x)              # 4 conv layers before GAP
        x = self.gap(x)                 # -> 1x1
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x