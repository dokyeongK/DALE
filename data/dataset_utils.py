import random
import PIL.Image as Image
from PIL import ImageEnhance

def get_patch_low_light(low_light, ground_truth, patch_size):
    #수정해야함
    height, width = low_light.size[1], low_light.size[0]

    ix = random.randrange(0, width - patch_size + 1)
    iy = random.randrange(0, height - patch_size + 1)

    # 가로시작점, 세로시작점, 가로범위, 세로범위
    crop_area = (ix, iy, ix + patch_size, iy + patch_size)

    low_light_img = low_light.crop(crop_area)
    ground_truth_img = ground_truth.crop(crop_area)

    return low_light_img, ground_truth_img

def get_patch_low_light_global(low_light, ground_truth, patch_size):
    #수정해야함
    height, width = low_light.size[1], low_light.size[0]

    ix = random.randrange(0, width - patch_size + 1)
    iy = random.randrange(0, height - patch_size + 1)

    # 가로시작점, 세로시작점, 가로범위, 세로범위
    crop_area = (ix, iy, ix + patch_size, iy + patch_size)
    select_num = random.randint(1,2)
    if select_num == 1:
        low_light_img = low_light.crop(crop_area)
        ground_truth_img = ground_truth.crop(crop_area)
    else :
        illumination =  random.randint(1,10) * 0.1
        low_light_img = ground_truth.crop(crop_area)
        global_illumination_Image = ImageEnhance.Brightness(low_light_img)
        low_light_img = global_illumination_Image.enhance(illumination)
        ground_truth_img = ground_truth.crop(crop_area)

    return low_light_img, ground_truth_img


def augmentation_low_light(low_light, ground_truth, args):
    rotate = args.augment_rotate == 0 and random.random() < 0.5
    augment_T2B = args.augment_T2B == 0 and random.random() < 0.5
    augment_L2R = args.augment_L2R == 0 and random.random() < 0.5

    if rotate :
        i = random.randint(0, 3)  # 1부터 100 사이의 임의의 정수
        rotate_list = [90,180,-90,-180]
        low_light = low_light.rotate(rotate_list[i])
        ground_truth = ground_truth.rotate(rotate_list[i])


    if augment_T2B:
        low_light = low_light.transpose(Image.FLIP_TOP_BOTTOM)
        ground_truth = ground_truth.transpose(Image.FLIP_TOP_BOTTOM)

    if augment_L2R:
        low_light = low_light.transpose(Image.FLIP_LEFT_RIGHT)
        ground_truth = ground_truth.transpose(Image.FLIP_LEFT_RIGHT)

    return low_light, ground_truth

def get_patch_sr(low_light_image, low_light_ground_truth_image, hr_image, patch_size, scale):

    hr_height, hr_width = hr_image.size[1], hr_image.size[0]
    lr_height, lr_width = low_light_image.size[1], low_light_image.size[0]

    hr_patch_size = (int)(scale * patch_size) # 128
    lr_patch_size = patch_size

    lr_x = random.randrange(0, lr_width - lr_patch_size + 1)
    lr_y = random.randrange(0, lr_height - lr_patch_size + 1)

    hr_x = lr_x * scale
    hr_y = lr_y * scale

    target_hr_x = random.randrange(0, hr_width - hr_patch_size + 1)
    target_hr_y = random.randrange(0, hr_height - hr_patch_size + 1)

    lr_crop_area = (lr_x, lr_y, lr_x+lr_patch_size, lr_y+lr_patch_size)
    hr_crop_area = (hr_x, hr_y, hr_x+hr_patch_size, hr_y+hr_patch_size)
    target_hr_crop_area = (target_hr_x, target_hr_y, target_hr_x + hr_patch_size, target_hr_y + hr_patch_size)

    lr_patch = low_light_image.crop(lr_crop_area)
    lr_gt_patch = low_light_ground_truth_image.crop(lr_crop_area)
    hr_patch = hr_image.crop(hr_crop_area)
    # lr_patch = hr_patch.resize((patch_size, patch_size), Image.BICUBIC)
    hr_target_patch = hr_image.crop(target_hr_crop_area)

    return lr_patch, lr_gt_patch, hr_patch, hr_target_patch

def augmentation_sr(lr_patch, lr_gt_patch, hr_patch, hr_target_patch, args):
    rotate = args.augment_rotate == 0 and random.random() < 0.5
    augment_T2B = args.augment_T2B == 0 and random.random() < 0.5
    augment_L2R = args.augment_L2R == 0 and random.random() < 0.5

    if rotate :
        i = random.randint(0, 3)  # 1부터 100 사이의 임의의 정수
        rotate_list = [90,180,-90,-180]
        lr_patch = lr_patch.rotate(rotate_list[i])
        lr_gt_patch = lr_gt_patch.rotate(rotate_list[i])
        hr_patch = hr_patch.rotate(rotate_list[i])
        hr_target_patch = hr_target_patch.rotate(rotate_list[i])


    if augment_T2B:
        lr_patch = lr_patch.transpose(Image.FLIP_TOP_BOTTOM)
        lr_gt_patch = lr_gt_patch.transpose(Image.FLIP_TOP_BOTTOM)
        hr_patch = hr_patch.transpose(Image.FLIP_TOP_BOTTOM)
        hr_target_patch = hr_target_patch.transpose(Image.FLIP_TOP_BOTTOM)

    if augment_L2R:
        lr_patch = lr_patch.transpose(Image.FLIP_LEFT_RIGHT)
        lr_gt_patch = lr_gt_patch.transpose(Image.FLIP_LEFT_RIGHT)
        hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)
        hr_target_patch = hr_target_patch.transpose(Image.FLIP_LEFT_RIGHT)

    return lr_patch, lr_gt_patch, hr_patch, hr_target_patch