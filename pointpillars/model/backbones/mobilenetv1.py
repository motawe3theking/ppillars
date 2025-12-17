import torch
import torch.nn as nn

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        # depthwise
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        # pointwise
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

class MobileNetV1(nn.Module):
    def __init__(self, in_channels=32):
        super(MobileNetV1, self).__init__()
        # Modified architecture: 32 -> 32 -> 64 -> 128
        # Stage 1: 32 input -> 32 output (stride 2, then 2x stride 1)
        self.stage1 = nn.Sequential(
            conv_dw(in_channels, 64, 2),  # 32 -> 64, stride 2
            conv_dw(64, 64, 1),            # 64 -> 64, stride 1
            conv_dw(64, 64, 1)             # 64 -> 64, stride 1
        )
        # Stage 2: 32 -> 64 (stride 2, then stride 1)
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),   # 64 -> 128, stride 2
            conv_dw(128, 128, 1)    # 128 -> 128, stride 1
        )
        # Stage 3: 64 -> 128 (stride 2, then multiple stride 1)
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),   # 64 -> 128, stride 2
            conv_dw(256, 256, 1),  # 128 -> 128, stride 1
            conv_dw(256, 256, 1),  # 128 -> 128, stride 1
            conv_dw(256, 256, 1),  # 128 -> 128, stride 1
            conv_dw(256, 256, 1),  # 128 -> 128, stride 1
            conv_dw(256, 256, 1)   # 128 -> 128, stride 1
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Returns intermediate features for feature fusion
        """
        x1 = self.stage1(x)      # 1/2
        x2 = self.stage2(x1)     # 1/4
        x3 = self.stage3(x2)     # 1/8
        
        return [x1, x2, x3]