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
from torch.optim import lr_scheduler

# Setting Loss #
L2_loss = nn.MSELoss().cuda()
Perceptual_loss = ploss.PerceptualLoss().cuda()
TvLoss = tvloss.TVLoss().cuda()

# Setting Visdom #
visdom = visdom.Visdom(env="DALE_EN")
loss_data = {'X': [], 'Y': [], 'legend_U':['mse_loss','tv_loss', 'p_loss']}

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
    for idx, key in enumerate(img_dict):
        win = window + idx
        tensor_img = train_utils.tensor2im(img_dict[key].data)
        visdom.image(tensor_img.transpose([2, 0, 1]), opts=dict(title=key), win=win)

def test(args, loader_test, model_AttentionNet, epoch, root_dir) :
    model_AttentionNet.eval()
    for itr, data in enumerate(loader_test):
        testImg, fileName = data[0], data[1]
        if args.cuda:
            testImg = testImg.cuda()

        with torch.no_grad():
            test_result = model_AttentionNet(testImg)
            test_result_img = train_utils.tensor2im(test_result)
            result_save_dir = root_dir + fileName[0].split('.')[0]+('_epoch_{}_itr_{}.png'.format(epoch, itr))
            train_utils.save_images(test_result_img, result_save_dir)

def main(args) :
    args.cuda = True
    args.epochs = 200
    args.lr = 1e-5
    args.batch_size = 4

    # Setting Important Path #
    train_data_root = 'D:\data\DALE/TRAIN/'
    model_save_root_dir = '../checkpoint/DALE/'
    model_root = '../checkpoint/'

    # Setting Important Traning Variable #
    VISUALIZATION_STEP = 50
    SAVE_STEP = 1

    print("DALE => Data Loading")

    train_data = dataset_DALE.DALETrain(train_data_root, args)
    loader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    print("DALE => Model Building")
    VisualAttentionNet =  VisualAttentionNetwork.VisualAttentionNetwork()

    state_dict = torch.load(model_root+'VAN.pth')
    VisualAttentionNet.load_state_dict(state_dict)

    EnhanceNet = EnhancementNet.EnhancementNet()

    print("DALE => Set Optimization")
    optG = torch.optim.Adam(list(EnhanceNet.parameters()), lr=args.lr, betas=(0.5, 0.999))

    scheduler = lr_scheduler.ExponentialLR(optG, gamma=0.99)

    model_EnhanceNet_parameters = filter(lambda p: p.requires_grad, EnhanceNet.parameters())

    params1 = sum([np.prod(p.size()) for p in model_EnhanceNet_parameters])

    print("Parameters | ", params1)

    print("DALE => Setting GPU")
    if args.cuda:
        print("DALE => Use GPU")
        VisualAttentionNet = VisualAttentionNet.cuda()
        EnhanceNet = EnhanceNet.cuda()
    print("DALE => Training")

    loss_step = 0

    for epoch in range(1, args.epochs):

        EnhanceNet.train()

        for itr, data in enumerate(loader_train):
            low_light_img, ground_truth_img, gt_Attention_img, file_name = data[0], data[1], data[2], data[3]
            if args.cuda:
                low_light_img = low_light_img.cuda()
                ground_truth_img = ground_truth_img.cuda()
                gt_Attention_img = gt_Attention_img.cuda()

            optG.zero_grad()

            attention_result = VisualAttentionNet(low_light_img)
            enhance_result = EnhanceNet(low_light_img, attention_result.detach())

            mse_loss = L2_loss(enhance_result, ground_truth_img)
            p_loss = Perceptual_loss(enhance_result, ground_truth_img) * 50
            tv_loss = TvLoss(enhance_result) * 20

            total_loss = p_loss + mse_loss + tv_loss

            total_loss.backward()
            optG.step()

            if epoch > 100 and itr==0:
                scheduler.step()
                print(scheduler.get_last_lr())

            if itr != 0 and itr % VISUALIZATION_STEP == 0:
                print("Epoch[{}/{}]({}/{}): "
                      "mse_loss : {:.6f}, "
                      "tv_loss : {:.6f}, "
                      "p_loss : {:.6f}"\
                      .format(epoch, args.epochs, itr, len(loader_train), mse_loss, tv_loss, p_loss))
                # VISDOM LOSS GRAPH #

                loss_dict = {
                    'mse_loss': mse_loss.item(),
                    'tv_loss' : tv_loss.item(),
                    'p_loss': p_loss.item(),
                }

                visdom_loss(visdom, loss_step, loss_dict)

                # VISDOM VISUALIZATION # -> tensor to numpy => list ('title_name', img_tensor)
                with torch.no_grad():
                    val_image = Image.open('../validation/15.jpg')

                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

                    val_image = transform((val_image)).unsqueeze(0)

                    val_image = val_image.cuda()

                    val_attention = VisualAttentionNet.eval()(val_image)
                    val_result = EnhanceNet.eval()(val_image, val_attention)

                img_list = OrderedDict(
                    [('input', low_light_img),
                     ('output', enhance_result),
                     ('attention_output', attention_result),
                     ('gt_Attention_img', gt_Attention_img),
                     ('batch_sum', attention_result+low_light_img),
                     ('ground_truth', ground_truth_img),
                     ('val_result', val_result)])

                visdom_image(img_dict=img_list, window=10)

                loss_step = loss_step + 1

        print("DALE => Testing")
        if epoch % SAVE_STEP == 0:
            train_utils.save_checkpoint(EnhanceNet, epoch, model_save_root_dir)

if __name__ == "__main__":
    opt = args
    main(opt)