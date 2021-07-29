import torch
import torch.utils.data
import numpy as np

# define the data process function (clipping the sound data)
class ProcessSoundData(torch.utils.data.Dataset):

    def __init__(self, X, Y, offset, context):

        # Add data and label to self
        self.X = X
        self.Y = Y

        # Define data index mapping
        index_map_X = []
        for i, x in enumerate(X):
            for j, xx in enumerate(x):
                index_pair_X = (i, j)
                index_map_X.append(index_pair_X)

        # Define label index mapping
        index_map_Y = []
        for i, y in enumerate(Y):
            for j, yy in enumerate(y):
                index_pair_Y = (i, j)
                index_map_Y.append(index_pair_Y)

        # Assert the data index mapping and label index mapping are the same
        assert (set(index_map_X) == set(index_map_Y))

        # Assign data index mapping to self
        self.index_map = np.array(index_map_X)

        # Add length to self
        self.length = (self.index_map).shape[0]

        # Add context and offset to self
        self.context = context
        self.offset = offset

        # Zero pad data as-needed for context size = 1
        for i, x in enumerate(self.X):
            self.X[i] = np.pad(x, ((self.offset, self.offset), (0, 0)), 'constant', constant_values=0)

    def __len__(self):

        # Return length
        return self.length

    def __getitem__(self, index):

        # Get index pair from index map
        i, j = self.index_map[index]

        # Calculate starting timestep using offset and context
        start_j = j + self.offset - self.context

        # Calculate ending timestep using offset and context
        end_j = j + self.offset + self.context + 1

        # Get data at index pair with context
        xx = self.X[i][start_j:end_j, :]

        # Get label at index pair
        yy = self.Y[i][j]

        # Return data at index pair with context and label at index pair
        return xx.ravel(), yy

    def collate_fn(batch):

        # Select all data from batch
        batch_x = [x for x, y in batch]

        # Select all labels from batch
        batch_y = [y for x, y in batch]

        # Convert batched data and labels to tensors
        batch_x = torch.as_tensor(batch_x).type(torch.FloatTensor)
        batch_y = torch.as_tensor(batch_y).type(torch.LongTensor)

        # Return batched data and labels
        return batch_x, batch_y