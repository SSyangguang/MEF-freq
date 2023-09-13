import argparse

parser = argparse.ArgumentParser(description='Fusion')
# Seed
parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--device', type=str, default='cuda:0',
                    help='gpu number')

parser.add_argument('--devices', type=int, default=[0, 1, 2, 3],
                    nargs='+',
                    help='gpu number')

# Data Acquisition
# parser.add_argument('--data_path', type=str, default='/data/yg/data/MEF_revise/Dataset_Part1',
#                     help='MEF dataset path')
parser.add_argument('--data_path', type=str, default='/data/yg/data/PQA-MEF/datasets/training_set',
                    help='MEF simple dataset path')
# parser.add_argument('--data_path', type=str, default='E:\project\data/MEF_revise/Dataset_Part1',
#                     help='MEF dataset path')
parser.add_argument('--data_test_path', type=str, default='/data/yg/data/PQA-MEF/datasets/test_set',
                    help='MEF dataset path')

# Training
parser.add_argument('--batch_size', type=int, default=100, help='batch size of fusion training')
parser.add_argument('--patch', type=int, default=256, help='patch size of fusion training')
parser.add_argument('--epochs', type=int, default=600, help='epochs of fusion training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of fusion training')
parser.add_argument('--wd', type=float, default=9e-3, help='weight decay of fusion training')
parser.add_argument('--eta', type=float, default=0.5, help='eta in en and std normalization')

# Test
parser.add_argument('--result_path', type=str, default='./result', help='file path of test results')
parser.add_argument('--resize', type=float, default=1, help='resize the test image to avoid cuda memory out')
parser.add_argument('--tau', type=int, default=128, help='tau value for the color channel fusion')
parser.add_argument('--block_size', type=int, default=800, help='block size for the test')

# Loss
parser.add_argument('--loss_alpha', type=int, default=0.5, help='alpha value for fusion model')

# Log file
parser.add_argument('--trans_log_dir', type=str, default='./trans_train_log', help='translation training log file path')
parser.add_argument('--trans_model_path', type=str, default='./trans_model/', help='translation model path')
parser.add_argument('--log_dir', type=str, default='./fusion_train_log', help='fusion training log file path')
parser.add_argument('--model_path', type=str, default='./fusion_model/', help='fusion model path')
parser.add_argument('--trans_model', type=str, default='trans_model.pth', help='translation model name')
parser.add_argument('--model', type=str, default='fusion_model.pth', help='fusion model name')
parser.add_argument('--feat_num', type=int, default=8, help='number of features')

args = parser.parse_args(args=[])
