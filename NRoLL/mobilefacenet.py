import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import math
from metric import mix_fc, cos_value

MobileFaceNet_BottleNeck_Setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]

def l2_norm(x, axis=1):
    norm = torch.norm(x, 2, axis, True)
    output = x / norm
    return output

class BottleNeck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(BottleNeck, self).__init__()
        self.connect = stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # 1*1 conv
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            # 3*3 depth wise conv
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            # 1*1 conv
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)

        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


class MobileFaceNet(nn.Module):
    def __init__(self, feature_dim=512, bottleneck_setting=MobileFaceNet_BottleNeck_Setting):
        super(MobileFaceNet, self).__init__()
        self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.cur_channel = 64
        block = BottleNeck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(128, 512, 1, 1, 0)
        self.linear7 = ConvBlock(512, 512, 8, 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, feature_dim, 1, 1, 0, linear=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.cur_channel, c, s, t))
                else:
                    layers.append(block(self.cur_channel, c, 1, t))
                self.cur_channel = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        return l2_norm(x)

class EncoderPlusArc(nn.Module):
    def __init__(self, feat_dim, num_class):
        super(EncoderPlusArc, self).__init__()
        self.encoder = MobileFaceNet(feat_dim)
        self.Parameter = Parameter(torch.Tensor(feat_dim, num_class))
        self.Parameter.requires_grad = True
        self.Parameter.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, input, label=None):
        feat_norm = self.encoder(input)
        if label == None:
            logists = cos_value(feat_norm, self.Parameter)
        else:
            logists = mix_fc(1.0, feat_norm, self.Parameter, label.detach(), fc_type='arcfc', margin=0.5, easy_margin=True)

        return logists
