from model import VisualAttentionNetwork, EnhancementNet
from data import dataset_DALE
from train import train_utils
import torch
from torch.utils.data import DataLoader

def main():
    benchmark = ['datasets__DICM', 'datasets__LIME', 'datasets__MEF', 'datasets__NPE']
    test_data_root = 'D:\data\DALE/benchmark/'+benchmark[0]
    model_root = '../checkpoint/'
    test_result_root_dir = '../EN_TEST/'

    VAN = VisualAttentionNetwork.VisualAttentionNetwork()
    state_dict1 = torch.load(model_root + 'VAN.pth')
    VAN.load_state_dict(state_dict1)

    EN = EnhancementNet.EnhancementNet()
    state_dict2 = torch.load(model_root + 'EN.pth')
    EN.load_state_dict(state_dict2)

    test_data = dataset_DALE.DALETest(test_data_root)
    loader_test = DataLoader(test_data, batch_size=1, shuffle=False)

    VAN.cuda()
    EN.cuda()

    test(loader_test, VAN, EN, test_result_root_dir)

def test(loader_test, VAN, EN, root_dir):
    VAN.eval()
    EN.eval()

    for itr, data in enumerate(loader_test):
        testImg, img_name = data[0], data[1]
        testImg = testImg.cuda()

        with torch.no_grad():
            visual_attention_map = VAN(testImg)
            enhance_result = EN(testImg, visual_attention_map)
            enhance_result_img = train_utils.tensor2im(enhance_result)
            result_save_dir = root_dir + 'enhance'+ img_name[0].split('.')[0]+('.png')
            train_utils.save_images(enhance_result_img, result_save_dir)

if __name__ == "__main__":
    main()