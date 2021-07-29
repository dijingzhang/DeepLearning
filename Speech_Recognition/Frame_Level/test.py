import torch
import os

def test(loader_test, model, device):
    predicted_list = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader_test, 0):
            x_batch, _ = data
            x_batch = x_batch.type(torch.FloatTensor)
            x_batch = x_batch.to(device)

            prediction_test = model(x_batch).to(device)
            _, predicted = torch.max(prediction_test.data, 1)
            predicted_list.append(predicted)

    print("Testing is done")
    return predicted_list