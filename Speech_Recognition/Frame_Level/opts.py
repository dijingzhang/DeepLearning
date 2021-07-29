import argparse


def get_opts():
    parser = argparse.ArgumentParser(description='Frame Level Classification of Speech')

    # Data Path
    parser.add_argument('--train_features_folder', type=str, default='train.npy')
    parser.add_argument('--train_labels_folder', type=str, default='train_labels.npy')
    parser.add_argument('--val_features_folder', type=str, default='val.npy')
    parser.add_argument('--val_labels_folder', type=str, default='val_labels.npy')
    parser.add_argument('--test_features_folder', type=str, default='test.npy')
    parser.add_argument('--model_folder', type=str, default='./model/')
    parser.add_argument('--models_list', type=list, default=[], help='models status you choose to ensemble')

    # Hyperparameters
    parser.add_argument('--max_iters', type=int, default=100, help='the number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=100, help='the number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='the number of epochs for training')

    parser.add_argument('--context', type=int, default=35)
    parser.add_argument('--offset', type=int, default=35)
    parser.add_argument('--frequency', type=int, default=40, help='the frequency of utterances')

    opts = parser.parse_args()

    return opts