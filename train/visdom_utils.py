import torchvision
import numpy as np

def tensor2im(image_tensor, imtype=np.uint8):
    # image_tensor = torch.clamp(image_tensor, min=0.0, max=1.0)
    image_numpy = torchvision.utils.make_grid(image_tensor).detach().cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))*255
    image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy.astype(imtype)

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
        tensor_img = tensor2im(img_dict[key].data)
        visdom.image(tensor_img.transpose([2, 0, 1]), opts=dict(title=key), win=win)