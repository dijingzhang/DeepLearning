import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import os

def train(model, train_dataset, val_dataset, batch_size, epochs, learning_rate, w_decay, device, model_folder):
    loader_train = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    loader_val = data.DataLoader(val_dataset, batch_size, num_workers=4)

    model.to(device)

    # loss function
    loss_func = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=w_decay, momentum=0.5)
    # LR scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2)

    # initialize several output
    acc_train = 0
    acc_val = 0

    # train loop
    for itr in range(epochs):
        loss = 0
        model.train()
        batch_num = 0
        for images, labels in loader_train:
            images = images.to(device)
            labels = labels.to(device)
            # adjsut the learning rate
            optimizer.zero_grad()
            out = model(images).to(device)
            loss_batch = loss_func(out, labels)
            loss += loss_batch

            loss_batch.backward()
            optimizer.step()

            # compute the accuracy over one batch
            _, predicted = torch.max(out.data, 1)

            acc_batch = (predicted == labels).sum().item() / batch_size
            acc_train += acc_batch

            # compute the total batch number
            batch_num += 1
        acc_train /= batch_num
        print('TRAIN: #{} epoch, the loss is '.format(i), loss / batch_num)
        print('TRAIN: #{} epoch, the acc_train is '.format(i), acc_train)

        # do validation
        batch_num_val = 0
        loss_val = 0
        model.eval()
        with torch.no_grad():
            for images_val, labels_val in loader_val:
                images_val = images_val.to(device)
                labels_val = labels_val.to(device)
                out_val = model.forward(images_val).to(device)
                _, predicted_val = torch.max(out_val.data, 1)
                loss_val += loss_func(out_val, labels_val)
                acc_batch_val = (predicted_val == labels_val).sum().item() / batch_size
                acc_val += acc_batch_val
                batch_num_val += 1
        acc_val /= batch_num_val
        loss_val /= batch_num_val
        lr_scheduler.step(loss_val)
        print('VAL: #{} epoch, the loss is '.format(itr), loss_val)
        print('VAL: #{} epoch, the acc_val is '.format(itr), acc_val)

        output_model_path = os.path.join(model_folder + 'model_state_' + str(itr) + '.pkl')
        torch.save(model.state_dict(), output_model_path)

    return
