import torch.nn as nn
from collections import OrderedDict

class Model(nn.Module):
    def __init__(self, context, frequency):
        super(Model, self).__init__()
        self.layer_dict = OrderedDict([
            ('linear1', nn.Linear(frequency * (2 * context + 1), 3550)),
            ('relu', nn.LeakyReLU(inplace=True)),
            ('bn1', nn.BatchNorm1d(3550)),
            ('dp1', nn.Dropout(0.25)),

            ('linear2', nn.Linear(3550, 3150)),
            ('relu2', nn.LeakyReLU(inplace=True)),
            ('bn2', nn.BatchNorm1d(3150)),
            ('dp2', nn.Dropout(0.27)),

            ('linear3', nn.Linear(3150, 2800)),
            ('relu3', nn.LeakyReLU(inplace=True)),
            ('bn3', nn.BatchNorm1d(2800)),
            ('dp3', nn.Dropout(0.3)),

            ('linear4', nn.Linear(2800, 2500)),
            ('relu4', nn.LeakyReLU(inplace=True)),
            ('bn4', nn.BatchNorm1d(2500)),
            ('dp4', nn.Dropout(0.3)),

            ('linear5', nn.Linear(2500, 2000)),
            ('relu5', nn.LeakyReLU(inplace=True)),
            ('bn5', nn.BatchNorm1d(2000)),
            ('dp5', nn.Dropout(0.3)),

            ('linear6', nn.Linear(2000, 2000)),
            ('relu6', nn.LeakyReLU(inplace=True)),
            ('bn6', nn.BatchNorm1d(2000)),
            ('dp6', nn.Dropout(0.3)),

            ('linear7', nn.Linear(2000, 1650)),
            ('relu7', nn.LeakyReLU(inplace=True)),
            ('bn7', nn.BatchNorm1d(1650)),
            ('dp7', nn.Dropout(0.3)),

            ('linear8', nn.Linear(1650, 1350)),
            ('relu8', nn.LeakyReLU(inplace=True)),
            ('bn8', nn.BatchNorm1d(1350)),
            ('dp8', nn.Dropout(0.3)),

            ('linear9', nn.Linear(1350, 1000)),
            ('relu9', nn.LeakyReLU(inplace=True)),
            ('bn9', nn.BatchNorm1d(1000)),
            ('dp9', nn.Dropout(0.27)),

            ('linear10', nn.Linear(1000, 650)),
            ('relu10', nn.LeakyReLU(inplace=True)),
            ('bn10', nn.BatchNorm1d(650)),
            ('dp10', nn.Dropout(0.25)),

            ('linear11', nn.Linear(650, 71)),
        ])
        self.layers = nn.Sequential(self.layer_dict)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output