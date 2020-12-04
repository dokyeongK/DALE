import argparse

parser = argparse.ArgumentParser(description='LessNet')

# Running Option

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Path Setting #####Add Part
parser.add_argument('--pre_model_dir', type=str, default='./pretrain_model/',
                    help='model directory')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=3,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='enables CUDA training')
parser.add_argument('--n_GPUs', type=int, default=0,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='./dataset',
                    help='dataset directory')
parser.add_argument('--data_custom', type=str, default='custom',
                    help='data custom check')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='MyImage',
                    help='test dataset name')
parser.add_argument('--n_train', type=int, default=1800,
                    help='number of training set')
parser.add_argument('--n_val', type=int, default=10,
                    help='number of validation set')
parser.add_argument('--offset_val', type=int, default=800,
                    help='validation index offest')
parser.add_argument('--offset_train', type=int, default=0,
                    help='validation index offest')
parser.add_argument('--ext', type=str, default='img',
                    help='dataset file extension')
#USE
parser.add_argument('--scale', type=int, default='4',
                    help='super resolution scale')
#USE
parser.add_argument('--patch_size', type=int, default=514,
                    help='output patch size')
#USE
parser.add_argument('--augment_rotate', type=int, default=0,
                    help='rotate option [ Yes -> 0, No -> 1')
#USE
parser.add_argument('--augment_T2B', type=int, default=0,
                    help='rotate option (flip Top to bottom) [ Yes -> 0, No -> 1')
#USE
parser.add_argument('--augment_L2R', type=int, default=0,
                    help='rotate option (flip Left to Right) [ Yes -> 0, No -> 1')

parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--quality', type=str, default='',
                    help='jpeg compression quality')
parser.add_argument('--chop_forward', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--testpath', type=str, default='./LR/LRBI/',#test/DIV2K_val_LR_our',
                    help='dataset directory for testing')
parser.add_argument('--testset', type=str, default='B100',
                    help='dataset name for testing')

# Model specifications
parser.add_argument('--model', default='EDSR',
                    help='model name')
parser.add_argument('--noise', type=str, default='.',
                    help='Gaussian noise std.')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')

# options for residual group and feature channel reduction
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=280,
                    help='do test per every N batches')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--resume', type=int, default=-1,
                    help='load the model from the specified epoch')
parser.add_argument('--batch_size', type=int, default=4,
                    help='input batch size for training')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')
parser.add_argument('--unfair', action='store_true',
                    help='select unfair option')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=1,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=int, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False