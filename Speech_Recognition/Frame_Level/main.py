import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from opts import get_opts
from train import train
from test import test
from ensemble import self_ensemble
from data_process import ProcessSoundData
from model import Model

def main():
    opts = get_opts()

    # Load the data
    train_features= np.load(opts.train_features_features_folder, allow_pickle=True)
    train_labels = np.load(opts.train_features_labels_folder, allow_pickle=True)
    val = np.load(opts.val_features_folder, allow_pickle=True)
    val_labels = np.load(opts.val_labels_folder, allow_pickle=True)
    test_features = np.load(opts.test_features_folder, allow_pickle=True)
    test_labels = [np.zeros(test[i].shape[0]) for i in range(len(test_features))]  # Fake labels for convenience
    print("Data loading is done")

    # Check whether GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Define a model
    model = Model(context=opts.context, frequency=opts.frequency)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=2.5e-6)
    loss_func = torch.nn.CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25, 35, 45], gamma=0.2)

    # Define dataloader
    train_dataset = ProcessSoundData(train_features, train_labels, opts.offset, opts.context)
    val_dataset = ProcessSoundData(val, val_labels, opts.offset, opts.context)
    test_dataset = ProcessSoundData(test_features, test_labels, opts.offset, opts.context)
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, pin_memory=True)

    # Train
    for itr in range(opts.max_iters):
        train(train_loader, val_loader, optimizer, model, loss_func, opts.batch_size, itr, lr_scheduler, opts.model_foler, device)

    # Self ensemble
    model = self_ensemble(opts.models_list, model)

    # Test
    result = test(test_loader, model, device)

if __name__ == '__main__':
    main()