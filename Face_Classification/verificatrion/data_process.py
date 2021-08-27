import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image
import csv

class NetDataset(data.Dataset):
    def __init__(self, filedata, transform, test=False):
        super(NetDataset,self).__init__()
        self.filedata = filedata
        self.transform = transform
        self.test = test

    def __getitem__(self, item):
        if self.test:
            pair = self.filedata[item][0]
            path1, path2 = pair.split(' ', 2)
            image1 = Image.open(str(path1))
            image2 = Image.open(str(path2))
            img1 = self.transform(image1)
            img2 = self.transform(image2)
            return img1.type('torch.FloatTensor'), img2.type('torch.FloatTensor')
        else:
            pair = self.filedata[item][0]
            path1, path2, label = pair.split(' ', 3)
            image1 = Image.open(str(path1))
            image2 = Image.open(str(path2))
            label = torch.from_numpy(np.array(int(label))).type('torch.LongTensor')
            img1 = self.transform(image1)
            img2 = self.transform(image2)
            return img1.type('torch.FloatTensor'), img2.type('torch.FloatTensor'), label

    def __len__(self):
        return self.filedata.shape[0]

def createDataloader(csv_file, batch_size, test=False):
    filedata = np.array(pd.read_csv(csv_file, header=None))

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    transform = T.Compose([T.ToTensor(), normalize, ])

    dataset = NetDataset(filedata=filedata, transform=transform, test=test)
    dataloader = data.DataLoader(dataset, num_workers=4, batch_size=batch_size)
    return dataloader