import torch.nn as nn
import torch
import model.BasicBlocks as BasicBlocks
import torch.nn.functional as F

class VisualAttentionNetwork(nn.Module):
    def __init__(self):
        super(VisualAttentionNetwork, self).__init__()

        self.feature_num = 64

        self.res_input_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1)  # 6
        )

        self.res_encoder1 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            BasicBlocks.Residual_Block_New(64, 64, 3),
        )

        self.down1 = DownSample(64)

        self.res_encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 1),
            BasicBlocks.Residual_Block_New(128, 128, 2),
        )

        self.down2 = DownSample(128)

        self.res_encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, 1),
            BasicBlocks.Residual_Block_New(256, 256, 1),
        )

        self.res_decoder3 = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            BasicBlocks.Residual_Block_New(256, 256, 1),
        )
        self.up2 = UpSample(256)

        self.res_decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            BasicBlocks.Residual_Block_New(128, 128, 2),
        )
        self.up1 = UpSample(128)

        self.res_decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            BasicBlocks.Residual_Block_New(64, 64, 3),
        )

        self.res_final = nn.Conv2d(64, 3, 3, 1, 1)

        # self.AttentionNet = AttenteionNet()

    def forward(self, x, only_attention_output=False):
        res_input = self.res_input_conv(x)

        encoder1 = self.res_encoder1(res_input)
        encoder1_down = self.down1(encoder1)
        #
        encoder2 = self.res_encoder2(encoder1_down)
        encoder2_down = self.down2(encoder2)

        encoder3 = self.res_encoder3(encoder2_down)

        decoder3 = self.res_decoder3(encoder3) + encoder3
        decoder3 = self.up2(decoder3, output_size=encoder2.size())

        decoder2 = self.res_decoder2(decoder3) + encoder2
        decoder2 = self.up1(decoder2, output_size=encoder1.size())

        decoder1 = self.res_decoder1(decoder2) + encoder1

        output = self.res_final(decoder1)

        return output

class DownSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(DownSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2)
        # self.conv2 = nn.Conv2d(in_channels, stride*in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

        self.avg_pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out

# --- Upsampling block in GridDehazeNet  --- #
class UpSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(UpSample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride=stride, padding=1)
        # self.conv = nn.Conv2d(in_channels, in_channels // stride, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x, output_size):
        out = F.relu(self.deconv(x, output_size=output_size))
        out = F.relu(self.conv(out))
        return out

