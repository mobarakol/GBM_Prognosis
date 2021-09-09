
import torch
import torch.nn as nn
import torch.nn.functional as F
from octconv import *
# Vanilla UNet implemntation is adopted from https://github.com/milesial/Pytorch-UNet
# Vanilla Octave convolution implemntation is adopted from https://github.com/d-li14/octconv.pytorch/blob/master/octconv.py

class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel))
        self.spatial_se = nn.Conv2d(channel, 1, kernel_size=1,
                                    stride=1, padding=0, bias=False)

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = torch.sigmoid(self.channel_excitation(chn_se).view(bahs, chs, 1, 1))
        chn_se = torch.mul(x, chn_se)
        spa_se = torch.sigmoid(self.spatial_se(x))
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, groups=1,
                 base_width=64, alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False):
        super(double_conv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = nn.Sequential(

            Conv_BN_ACT(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, groups=groups, norm_layer=norm_layer,
                        alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5),
            Conv_BN_ACT(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, groups=groups, norm_layer=norm_layer,
                        alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False):
        super(inconv, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            Conv_BN_ACT(planes, width, kernel_size=3, alpha_in=0.5 if output else 0, alpha_out=alpha_out, norm_layer=norm_layer)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None, groups=1,
                 base_width=64, alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            OctMaxPool2d(2),
            double_conv(in_ch, out_ch, stride=1, downsample=None, groups=1,
                 base_width=64, alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = OctUp(scale_factor=2, size=(None, None))
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)
        self.SCSE = SCSEBlock(int(out_ch/2))
        self.upsize = OctUp_size()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.upsize(x1, x2)
        x_h = torch.cat([x2[0], x1[0]], dim=1)
        x_l = torch.cat([x2[1], x1[1]], dim=1)
        x = (x_h, x_l)
        x = self.conv(x)
        x_h = self.SCSE(x[0])
        x_l = self.SCSE(x[1])
        return (x_h, x_l)


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()

        self.conv = OctaveConv(in_ch, out_ch, kernel_size=1, alpha_in=0.5, alpha_out=0)

    def forward(self, x):
        x = self.conv(x)
        return x

class OctaveUNet(nn.Module):
    def __init__(self, n_classes, in_ch=3):
        super(OctaveUNet, self).__init__()
        self.inc = inconv(in_ch, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        #encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        #decoder
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        x10 = self.outc(x9)
        out = F.upsample(x10[0], (x.size(2), x.size(3) ), mode='bilinear')
        return out
