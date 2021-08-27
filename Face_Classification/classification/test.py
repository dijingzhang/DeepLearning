import torch
import torch.utils.data as data


def test(model, device, test_dataset):
    loader_test = data.DataLoader(test_dataset, 64)
    model.eval()
    predicted_list = []
    with torch.no_grad():
        for images_test, _ in loader_test:
            images_test = images_test.to(device)
            out_test = model.forward(images_test).to(device)
            _, predicted_test = torch.max(out_test.data, 1)
            predicted_list.append(predicted_test)

    print("Testing is done")
    return predicted_list