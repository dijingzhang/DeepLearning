import numpy as np
import torch
import pandas as pd

from opts import get_opts
from train import train
from test import test
from model import SEResNet50
from data_process import createDataset
from ensemble import self_ensemble

def main():
    opts = get_opts()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 8 if device != "cpu" else 0
    print("Now running device is: ", device, " with num of workers ", num_workers)

    train_dataset, val_dataset, test_dataset = createDataset(opts.train_path, opts.val_path, opts.test_path)

    senet = SEResNet50(n_classes=opts.n_classes)
    # train the model
    train(model=senet, train_dataset=train_dataset, val_dataset=val_dataset, batch_size=opts.batch_size,
                  epochs=opts.max_iters, learning_rate=opts.lr, w_decay=opts.w_decay, device=device)

    # self-ensemble
    senet = self_ensemble(opts.models_list, senet)

    predicted_test = test(model=senet, device=device, test_dataset=test_dataset)

    # use the test dataset to validate the model

    pred = []
    for i in range(len(predicted_test)):
        pred.append(np.array(predicted_test[i].cpu()).ravel())
    pred = np.array(pred).ravel()

    name = []
    for i in range(8000):
        name.append(str(i) + '.jpg')

    prediction = np.append(np.array(name).reshape((8000, 1)), pred.astype(np.int64).reshape((8000, 1)), axis=1)
    dataframe = pd.DataFrame({"id": prediction[:, 0], 'label': prediction[:, 1]})
    dataframe.to_csv("dijingz.csv", index=False, sep=',')


if __name__ == '__main__':
    main()