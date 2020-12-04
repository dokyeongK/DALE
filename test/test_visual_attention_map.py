from model import VisualAttentionNetwork
from data import dataset_DALE
from train import train_utils
import torch
from torch.utils.data import DataLoader

def main():

    # test_data_root = 'D:\data\LessNet\/enhance_test'
    test_data_root = 'D:\data\DarkPair\ExDark\Bicycle'
    model_root = '../checkpoint/'
    test_result_root_dir = '../VAN_TEST/'

    # model_LessNet = LessNet.LessNet(stride=1)
    VisualAttentioNNet =  VisualAttentionNetwork.VisualAttentionNetwork()#LessNet_Update.LessNet()#AttentionNet.AttenteionNet(stride=1)
    state_dict = torch.load(model_root+'VAN.pth')
    VisualAttentioNNet.load_state_dict(state_dict)

    test_data = dataset_DALE.DALETest(test_data_root)
    loader_test = DataLoader(test_data, batch_size=1, shuffle=False)

    VisualAttentioNNet.cuda()
    test(loader_test, VisualAttentioNNet, test_result_root_dir)

def test(loader_test, visualAttentionNet, root_dir) :
    visualAttentionNet.eval()
    for itr, data in enumerate(loader_test):
        testImg, fileName = data[0], data[1]
        testImg = testImg.cuda()

        with torch.no_grad():
            test_attention_result= visualAttentionNet(testImg)

            test_recon_result_img = train_utils.tensor2im(test_attention_result)
            norm_input_img = train_utils.tensor2im(testImg+test_attention_result)

            recon_save_dir = root_dir + 'visual_attention_map_'+fileName[0].split('.')[0]+('.png')
            recon_save_dir2 = root_dir + 'sum_'+fileName[0].split('.')[0]+('.png')

            train_utils.save_images(test_recon_result_img, recon_save_dir)
            train_utils.save_images(norm_input_img, recon_save_dir2)

if __name__ == '__main__':
    main()