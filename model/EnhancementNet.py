import torch.nn as nn
import torch
import torch.nn.utils.spectral_norm as spectral_norm
import model.BasicBlocks as BasicBlocks
import torch.nn.functional as F
import numpy as np

class EnhancementNet(nn.Module):
    def __init__(self):
        super(EnhancementNet, self).__init__()

        self.feature_num = 64

        self.res_input_conv = nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1) # 6
        )

        self.residual_group1 = BasicBlocks.Residual_Block_Enhance(64, 64, 3)

        self.residual_group2 = BasicBlocks.Residual_Block_Enhance(64, 64, 2)

        self.residual_group3 = BasicBlocks.Residual_Block_Enhance(64, 64, 1)

        self.se = BasicBlocks.SELayer(192)

        self.conv_block = nn.Sequential(
            nn.Conv2d(192, 64, 3, 1, 1),
            nn.Conv2d(64, 3, 3, 1, 1)
        )


        # self.res_final = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x, attention):

        res_input = self.res_input_conv(torch.cat([x,attention],1))
        res1 = self.residual_group1(res_input)
        res2 = self.residual_group2(res1)
        res3 = self.residual_group3(res2)
        group_cat = self.se(torch.cat([res1,res2,res3],1))

        output = self.conv_block(group_cat) + attention

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        num_input_channels = 3
        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        ndf = 48
        use_bias = False
        use_sigmoid = False

        norm_layer = spectral_norm  # spectral_norm# nn.InstanceNorm2d(ndf)
        # norm_layer = nn.InstanceNorm2d

        sequence = [norm_layer(nn.Conv2d(num_input_channels, ndf, kernel_size=kw, stride=2, padding=padw)),
                    nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        n_layers =2  ########

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                norm_layer(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                     kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            norm_layer(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                 kernel_size=kw, stride=1, padding=padw, bias=use_bias)),

            nn.LeakyReLU(0.2, True)
        ]

        sequence += [norm_layer(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        # features = self.features(x)
        # output = self.classifier(features.view(features.size(0), -1))
        output = self.model(x)
        return output
