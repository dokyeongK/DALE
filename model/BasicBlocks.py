import torch.nn as nn
import torch
import torch.nn.functional as F

class _Conv_Block(nn.Module):
    def __init__(self):
        super(_Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.BatchNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.BatchNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        return output

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class SALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True), #####LeakyReLU
                nn.Conv2d(channel, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_du(x)
        return x * y

#SE BLOCK
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Residual_Block_New(nn.Module):
    def __init__(self, in_num, out_num, dilation_factor):
        super(Residual_Block_New, self).__init__()
        self.conv1 = (nn.Conv2d(in_channels=in_num, out_channels=out_num, kernel_size=3, stride=1, padding=dilation_factor, dilation=dilation_factor, groups= 1, bias=False))
        self.in1 = nn.BatchNorm2d(out_num)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = (nn.Conv2d(in_channels=out_num, out_channels=out_num, kernel_size=3, stride=1, padding=dilation_factor, dilation=dilation_factor, groups= 1, bias=False))
        self.in2 = nn.BatchNorm2d(out_num)
        self.se = SELayer(channel=out_num)

    def forward(self, x):
        identity_data = x
        output = self.relu((self.conv1(x)))
        output = (self.conv2(output))

        #USE SE BLOCK
        se = self.se(output)
        output = se + identity_data
        return output

class Residual_Block_Enhance(nn.Module):
    def __init__(self, in_num, out_num, dilation_factor):
        super(Residual_Block_Enhance, self).__init__()
        self.conv1 = (nn.Conv2d(in_channels=in_num, out_channels=out_num, kernel_size=3, stride=1, padding=dilation_factor, dilation=dilation_factor, groups= 1, bias=False))
        self.in1 = nn.BatchNorm2d(out_num)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = (nn.Conv2d(in_channels=out_num, out_channels=out_num, kernel_size=3, stride=1, padding=dilation_factor, dilation=dilation_factor, groups= 1, bias=False))
        self.in2 = nn.BatchNorm2d(out_num)
        self.se = SELayer(channel=out_num)

    def forward(self, x):
        identity_data = x
        output = self.relu((self.conv1(x)))
        output = (self.conv2(output))

        #USE SE BLOCK
        se = self.se(output)
        output = se + identity_data
        return output

class Residual_Block(nn.Module):
    def __init__(self):
        super(Residual_Block, self).__init__()
        self.conv1 = (nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, groups=1, bias=False))
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        # self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = (nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, groups=1, bias=False))
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.se = SELayer(channel=64)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        #USE SE BLOCK
        se = self.se(output)
        output = se + identity_data
        return output

class make_dense(nn.Module):
      def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
      def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
      def __init__(self, nChannels=64, nDenselayer=5, growthRate=16):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
        self.se = SELayer(nChannels)
      def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = (out) + x
        return out