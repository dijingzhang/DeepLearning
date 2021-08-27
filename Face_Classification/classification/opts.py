import argparse


def get_opts():
    parser = argparse.ArgumentParser(description='Frame Level Classification of Speech')

    # Data Path
    parser.add_argument('--train_path', type=str, default='train_data')
    parser.add_argument('--val_path', type=str, default='val_data')
    parser.add_argument('--test_path', type=str, default='test_data')

    parser.add_argument('--model_folder', type=str, default='./model/')
    parser.add_argument('--models_list', type=list, default=[], help='models status you choose to ensemble')

    parser.add_argument('--n_classes', type=int, default=4000, help='the number of classification categories')


    # Hyperparameters
    parser.add_argument('--max_iters', type=int, default=100, help='the number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=128, help='the number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--w_decay', type=float, default=1e-5, help='weight decay for optimizer')


    opts = parser.parse_args()

    return opts