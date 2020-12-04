import os
from torch.utils.data import Dataset
import PIL.Image as Image
from torchvision import transforms
from data import dataset_utils as utils
import numpy as np
import torch
# Class for train_lr
class DALETrain(Dataset):
    def __init__(self, root_dir, args):
        """
        Arguments:
            1) root directory -> D:\data\LessNet\TRAIN\
            2) arguments -> args
        """
        self.low_light_dir = root_dir + 'SuperPixel'
        self.ground_truth_dir = root_dir + 'GT'
        self.low_light_img_list = os.listdir(root_dir + 'SuperPixel')
        self.ground_truth_img_list = os.listdir(root_dir + 'GT')

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform2 = transforms.Compose([
            transforms.ToTensor()
        ])

        # patch_size : default == 128
        self.args = args
        self.patch_size = 240


    def __len__(self):
        return len(self.low_light_img_list)

    def __getitem__(self, idx):
        """
        Get a random pair of image crops.
        It returns a tuple of float tensors with shape [3, height, width].
        They represent RGB images with pixel values in [0, 1] range.
        :return: Low-light image, Ground-truth image tensor
        """
        low_light_name, ground_truth_name = self.low_light_img_list[idx], self.ground_truth_img_list[idx]

        low_light_image = Image.open(os.path.join(self.low_light_dir, low_light_name))
        ground_truth_image = Image.open(os.path.join(self.ground_truth_dir, ground_truth_name))

        # Get crop image
        low_light_patch, ground_truth_patch = utils.get_patch_low_light(low_light_image, ground_truth_image, self.patch_size)

        # Get augmented image
        low_light_patch, ground_truth_patch = utils.augmentation_low_light(low_light_patch, ground_truth_patch, self.args)
        # Get the image buffer as ndarray

        buffer1 = np.asarray(low_light_patch).astype(np.long)

        buffer2 = np.asarray(ground_truth_patch).astype(np.long)

        # Subtract image2 from image1

        attention_patch = np.clip(buffer2 - buffer1, 0, 255).astype(np.uint8)

        # Convert to tensor
        low_light_tensor = self.transform2(low_light_patch)
        ground_truth_tensor = self.transform2(ground_truth_patch)
        attention_tensor = self.transform2(attention_patch)


        return low_light_tensor, ground_truth_tensor, attention_tensor, ground_truth_name

class DALETrainGlobal(Dataset):
    def __init__(self, root_dir, args):
        """
        Arguments:
            1) root directory -> D:\data\LessNet\TRAIN\
            2) arguments -> args
        """
        self.low_light_dir = root_dir + 'SuperPixel'
        self.ground_truth_dir = root_dir + 'GT'
        self.low_light_img_list = os.listdir(root_dir + 'SuperPixel')
        self.ground_truth_img_list = os.listdir(root_dir + 'GT')

        self.transform2 = transforms.Compose([
            transforms.ToTensor()
        ])

        # patch_size : default == 128
        self.args = args
        self.patch_size = 240


    def __len__(self):
        return len(self.low_light_img_list)

    def __getitem__(self, idx):

        low_light_name, ground_truth_name = self.low_light_img_list[idx], self.ground_truth_img_list[idx]

        low_light_image = Image.open(os.path.join(self.low_light_dir, low_light_name))
        ground_truth_image = Image.open(os.path.join(self.ground_truth_dir, ground_truth_name))

        # Get crop image
        low_light_patch, ground_truth_patch = utils.get_patch_low_light_global(low_light_image, ground_truth_image, self.patch_size)

        # Get augmented image
        low_light_patch, ground_truth_patch = utils.augmentation_low_light(low_light_patch, ground_truth_patch, self.args)
        # Get the image buffer as ndarray

        buffer1 = np.asarray(low_light_patch)

        buffer2 = np.asarray(ground_truth_patch)

        # Subtract image2 from image1

        attention_patch = buffer2 - buffer1

        # Convert to tensor
        low_light_tensor = self.transform2(low_light_patch)
        ground_truth_tensor = self.transform2(ground_truth_patch)
        attention_tensor = self.transform2(attention_patch)

        return low_light_tensor, ground_truth_tensor, attention_tensor, ground_truth_name

# Class for test
class DALETest(Dataset):
    def __init__(self, root_dir):
        """
        Arguments:
            1) root directory -> D:\data\LessNet\TEST\
            2) arguments -> args
        """
        self.root_dir = root_dir
        self.test_img_list = os.listdir(root_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.test_img_list)

    def __getitem__(self, idx):
        """
        Get a test image tensor.
        It returns a tuple of float tensors with shape [3, height, width].
        They represent RGB images with pixel values in [0, 1] range.
        :return: test image tensor, file name
        """
        test_img_name = self.test_img_list[idx]
        # Open Image
        test_image = Image.open(os.path.join(self.root_dir, test_img_name))
        test_image_tensor = self.transform(test_image)
        return test_image_tensor, test_img_name
