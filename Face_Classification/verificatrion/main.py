import torch

from opts import get_opts
from test import test
from model import Model
from data_process import createDataloader

def main():
    opts = get_opts()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 8 if device != "cpu" else 0
    print("Now running device is: ", device, " with num of workers ", num_workers)

    val_dataloader = createDataloader(csv_file=opts.val_path, batch_size=opts.batch_size, test=False)
    test_dataloader = createDataloader(csv_file=opts.test_path, batch_size=opts.batch_size, test=True)

    model = Model(opts.model_path)
    model.to(device)

    test(model, device, val_dataloader, test_path="", test=False)
    test(model, device, test_dataloader, test_path=opts.test_path, test=True)

if __name__ == '__main__':
    main()