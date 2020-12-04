import torch
import numpy as np
from PIL import Image
import torchvision
import scipy.misc as misc

def adjust_learning_rate(epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (opt.gamma ** ((epoch % opt.epochs) // opt.lr_decay))
    return lr

def save_checkpoint(model, epoch, root_dir): #model_out_path
    """
    :param path: model 저장 명 Pull Path
    """
    print("LessNet => Saving Model")
    model_out_path = "tunning_low_part_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), root_dir + model_out_path )

def save_images(img_numpy, img_path):
    """train 결과 이미지 저장 함수"""
    image_pil = None
    if img_numpy.shape[2] == 1:
        img_numpy = np.reshape(img_numpy, (img_numpy.shape[0], img_numpy.shape[1]))
        image_pil = Image.fromarray(img_numpy, 'L')
    else:
        image_pil = Image.fromarray(img_numpy)
    image_pil.save(img_path)

def tensor2im(image_tensor, imtype=np.uint8):
    # image_tensor = torch.clamp(image_tensor, min=0.0, max=1.0)
    image_numpy = torchvision.utils.make_grid(image_tensor).detach().cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))*255
    image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy.astype(imtype)

def save_results_RGB(result, filename):
    filename =  filename
    normalized = result[0].data.mul(255 / 255)
    ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
    misc.imsave('{}.png'.format(filename), ndarr)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
