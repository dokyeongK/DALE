from option import args
from data import dataset_DALE
from torch.utils.data import DataLoader
from model import VisualAttentionNetwork, EnhancementNet
from train import train_utils
from collections import OrderedDict
import numpy as np
from loss import ploss, tvloss
import visdom
import PIL.Image as Image
from torchvision import transforms

import torch.nn as nn
import torch

# Setting Loss #
L2_loss = nn.MSELoss().cuda()
Perceptual_loss = ploss.PerceptualLoss().cuda()
TvLoss = tvloss.TVLoss().cuda()

# Setting Visdom #
visdom = visdom.Visdom(env="DALEGAN")
loss_data = {'X': [], 'Y': [], 'legend_U':['e_loss','tv_loss', 'p_loss', 'g_loss', 'd_loss']}

def visdom_loss(visdom, loss_step, loss_dict):
    loss_data['X'].append(loss_step)
    loss_data['Y'].append([loss_dict[k] for k in loss_data['legend_U']])
    visdom.line(
        X=np.stack([np.array(loss_data['X'])] * len(loss_data['legend_U']), 1),
        Y=np.array(loss_data['Y']),
        win=1,
        opts=dict(xlabel='Step',
                  ylabel='Loss',
                  title='Training loss',
                  legend=loss_data['legend_U']),
        update='append'
    )

def visdom_image(img_dict, window):
    """VISDOM에 이미지들 띄우는 역할"""
    for idx, key in enumerate(img_dict):
        win = window + idx
        tensor_img = train_utils.tensor2im(img_dict[key].data)
        visdom.image(tensor_img.transpose([2, 0, 1]), opts=dict(title=key), win=win)

def test(args, loader_test, model_AttentionNet, epoch, root_dir) :
    """Not Implement Yet"""
    model_AttentionNet.eval()

    for itr, data in enumerate(loader_test):
        testImg, fileName = data[0], data[1]
        if args.cuda:
            testImg = testImg.cuda()

        with torch.no_grad():
            test_result = model_AttentionNet(testImg)

            # Normalization to origin image 필요!
            test_result_img = train_utils.tensor2im(test_result)

            result_save_dir = root_dir + fileName[0].split('.')[0]+('_epoch_{}_itr_{}.png'.format(epoch, itr))

            train_utils.save_images(test_result_img, result_save_dir)

def main(args) :
    """Main Function : Data Loading -> Model Building -> Set Optimization -> Training"""
    # Setting Important Arguments #
    args.cuda = True
    args.epochs = 200
    args.lr = 1e-5
    args.batch_size = 5
    # Setting Important Path #
    train_data_root = 'D:\data\DALE/'
    model_save_root_dir = 'D:\Pytorch_code\DALE/checkpoint/'
    model_root = '../checkpoint/DALEGAN/'

   # Setting Important Traning Variable #
    VISUALIZATION_STEP = 50
    SAVE_STEP = 1

    print("DALE => Data Loading")

    train_data = dataset_DALE.DALETrainGlobal(train_data_root, args)
    loader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    print("DALE => Model Building")
    VAN =  VisualAttentionNetwork.AttentionNet2()
    state_dict1 = torch.load(model_root+'visual_attention_network_model.pth')
    VAN.load_state_dict(state_dict1)

    EnhanceNetG = EnhancementNet.EnhancementNet()
    EnhanceNetD = EnhancementNet.Discriminator()

    state_dict2 = torch.load(model_root+'enhance_GAN.pth')
    EnhanceNetG.load_state_dict(state_dict2)

    EnhancementNet_parameters = filter(lambda p: p.requires_grad, EnhanceNetG.parameters())

    params1 = sum([np.prod(p.size()) for p in EnhancementNet_parameters])

    print("Parameters | Discriminator ", params1)

    discriminator_parameters = filter(lambda p: p.requires_grad, EnhanceNetD.parameters())
    params = sum([np.prod(p.size()) for p in discriminator_parameters])

    print("Parameters | Discriminator ", params)

    print("DALE => Set Optimization")
    optG = torch.optim.Adam(list(EnhanceNetG.parameters()), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(list(EnhanceNetD.parameters()), lr=args.lr, betas=(0.5, 0.999),
                            weight_decay=0)

    print("DALE => Setting GPU")
    if args.cuda:
        print("DALE => Use GPU")
        VAN = VAN.cuda()
        EnhanceNetG = EnhanceNetG.cuda()
        EnhanceNetD = EnhanceNetD.cuda()

    print("DALE => Training")

    loss_step = 0

    for epoch in range(args.epochs):

        EnhanceNetG.train()
        EnhanceNetD.train()
        for itr, data in enumerate(loader_train):
            low_light_img, ground_truth_img, gt_Attention_img, file_name = data[0], data[1], data[2], data[3]
            if args.cuda:
                low_light_img = low_light_img.cuda()
                ground_truth_img = ground_truth_img.cuda()
                gt_Attention_img = gt_Attention_img.cuda()

            optD.zero_grad()


            attention_result = VAN(low_light_img)
            enhance_result = EnhanceNetG(low_light_img, attention_result).detach()

            loss_D = -torch.mean(EnhanceNetD(ground_truth_img)) \
                     + torch.mean(EnhanceNetD(enhance_result))

            loss_D.backward()
            optD.step()

            for p in EnhanceNetD.parameters():
                p.data.clamp_(-0.01, 0.01)

            if itr % 5 == 0:

                optG.zero_grad()
                enhance_result = EnhanceNetG(low_light_img, attention_result)
                loss_G = -torch.mean(EnhanceNetG(enhance_result)) * 0.5

                e_loss = L2_loss(enhance_result, ground_truth_img)
                p_loss = Perceptual_loss(enhance_result, ground_truth_img) * 10
                tv_loss = TvLoss(enhance_result) * 5

                total_loss = p_loss + e_loss +tv_loss +loss_G

                total_loss.backward()
                optG.step()

            if itr != 0 and itr % VISUALIZATION_STEP == 0:
                print("Epoch[{}/{}]({}/{}): "
                      "e_loss : {:.6f}, "
                      "tv_loss : {:.6f}, "
                      "p_loss : {:.6f}"\
                      .format(epoch, args.epochs, itr, len(loader_train), e_loss,tv_loss, p_loss))

                # VISDOM LOSS GRAPH #

                loss_dict = {
                    'e_loss': e_loss.item(),
                    'tv_loss' : tv_loss.item(),
                    'p_loss': p_loss.item(),
                    'g_loss' : loss_G.item(),
                    'd_loss' : loss_D.item()
                    # 'recon_loss': recon_loss.item()
                }

                visdom_loss(visdom, loss_step, loss_dict)

                # VISDOM VISUALIZATION # -> tensor to numpy => list ('title_name', img_tensor)

                with torch.no_grad():
                    val_image = Image.open('../validation/15.jpg')

                    transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])

                    val_image = transform((val_image)).unsqueeze(0)

                    val_image = val_image.cuda()
                    val_attention = VAN.eval()(val_image)
                    val_result = EnhanceNetG.eval()(val_image, val_attention)

                img_list = OrderedDict(
                    [('input', low_light_img),
                     ('output', enhance_result),
                     ('attention_output', attention_result),
                     ('gt_Attention_img', gt_Attention_img),
                     ('ground_truth', ground_truth_img),
                     ('val_result', val_result),
                     ('val_sum', val_attention+val_image)])

                visdom_image(img_dict=img_list, window=10)

                loss_step = loss_step + 1

        print("DALE => Testing")

        if epoch % SAVE_STEP == 0:
            train_utils.save_checkpoint(EnhanceNetG, epoch, model_save_root_dir + 'DALEGAN/')
            train_utils.save_checkpoint(EnhanceNetD, epoch, model_save_root_dir + 'DALE_Discriminator/')

if __name__ == "__main__":
    opt = args
    main(opt)