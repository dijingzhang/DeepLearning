import torch
import numpy as np
import pandas as pd
import os
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image
from torchvision.datasets import ImageFolder
import csv

class MyImageFolder(ImageFolder):
    def _find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort(key=lambda x: int(x))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class FaceDataset(data.Dataset):
    def __init__(self, dataset):
        super(FaceDataset, self).__init__()
        self.dataset = dataset


    # read image and take item as index
    def __getitem__(self, item):
        face = self.dataset[item][0]
        label = torch.from_numpy(np.array(self.dataset[item][1])).type('torch.LongTensor')
        return face, label

    def __len__(self):
        return len(self.dataset)

# dataset loader for test data
class TestDataset(data.Dataset):
    def __init__(self, filename, transform):
        super(TestDataset, self).__init__()
        self.filename = filename
        self.transform = transform
        self.path = np.array(pd.read_csv(filename, header=None, usecols=[0]))
        self.label = np.array(pd.read_csv(filename, header=None, usecols=[1]))

    def __getitem__(self, item):
        image = Image.open(self.path[item][0])
        image_tensor = self.transform(image)
        label = torch.from_numpy(np.array(self.label[item][0])).type('torch.LongTensor')
        return image_tensor.type('torch.FloatTensor'), label

    def __len__(self):
        return self.path.shape[0]

# data augmentation
# Add Noise
# SaltPepperNoise
class AddSaltPepperNoise(object):
    def __init__(self, density=0.0):
        self.density = density

    def __call__(self, img):
        h, w, c = img.shape
        Std = 1.0 - self.density
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[self.density/2.0, self.density/2.0, Std])
        mask = np.repeat(mask, c, axis=2)
        mask = torch.from_numpy(mask)
        img[mask == 0] = 0
        img[mask == 1] = 1
        return img

# GaussianNoise
class AddGaussianNoise(object):
    def __init__(self, mean=0, var=1.0, amplitude=1.0):
        self.mean = mean
        self.var = var
        self.amplitude = amplitude

    def __call__(self, img):
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.var, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = torch.from_numpy(N).type('torch.FloatTensor') + img
        img[img > 1.0] = 1.0
        img[img < 0.0] = 0.0
        return img


def createDataset(train_path, val_path, test_path):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    transform_init = T.Compose([T.ToTensor(), normalize])
    transform_aug = T.Compose([T.ToTensor(), normalize, T.RandomHorizontalFlip(p=0.5), T.RandomGrayscale(p=0.2),
                               T.RandomRotation(degrees=10)])

    dataset_train = MyImageFolder(train_path, transform_aug)
    dataset_val = MyImageFolder(val_path, transform_init)

    # load the path of test dataset and create a fake label
    dataset_test = []
    for i in range(8000):  # there are 8,000 images
        path = test_path + str(i) + '.jpg'
        label = -1  # create fake label
        dataset_test.append([path, label])

    f = open('dataset_test.csv', 'a', newline='')
    for i in range(len(dataset_test)):
        writer = csv.writer(f)
        writer.writerow(dataset_test[i])
    f.close()

    train_dataset = FaceDataset(dataset_train)
    val_dataset = FaceDataset(dataset_val)
    test_dataset = TestDataset('dataset_test.csv')

    return train_dataset, val_dataset, test_dataset