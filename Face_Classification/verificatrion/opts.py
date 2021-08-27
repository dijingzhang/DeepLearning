import argparse


def get_opts():
    parser = argparse.ArgumentParser(description='Frame Level Classification of Speech')

    # Data Path
    parser.add_argument('--val_path', type=str, default='verification_pairs_val.txt')
    parser.add_argument('--test_path', type=str, default='verification_pairs_test.txt')

    parser.add_argument('--model_path', type=str, default='')


    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help='the number of epochs for training')


    opts = parser.parse_args()

    return opts