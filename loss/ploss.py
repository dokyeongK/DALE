import torch.nn as nn
from torchvision.models.vgg import vgg16, vgg19

#
# class PerceptualLoss(nn.Module):
#     def __init__(self):
#         super(PerceptualLoss, self).__init__()
#         vgg = vgg16(pretrained=True).cuda()
#         self.loss_network = nn.Sequential(*list(vgg.features)[:16]).eval()
#         for param in self.loss_network.parameters():
#             param.requires_grad = False
#         self.mse_loss = nn.MSELoss()
#         self.l1_loss = nn.L1Loss()
#
#     def forward(self, out_images, target_images):
#         loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
#         return loss



# --- Perceptual loss network  --- #
class PerceptualLoss(nn.Module):

    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg16(pretrained=True).cuda()
        self.loss_network = nn.Sequential(*list(vgg.features)[:16]).eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False

        self.l1_loss = nn.L1Loss()

    def normalize_batch(self, batch):
        # Normalize batch using ImageNet mean and std
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        return (batch - mean) / std

    def forward(self, out_images, target_images):

        loss = self.l1_loss(
            self.loss_network(self.normalize_batch(out_images)),
            self.loss_network(self.normalize_batch(target_images))
        )

        return loss
