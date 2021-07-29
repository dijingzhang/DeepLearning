import torch
import os


def train(train_loader, val_loader,optimizer, model, loss_func, batchsize, itr, lr_scheduler, model_foler, device):
    total_loss, total_acc, batch_num = 0, 0, 0

    # TRAIN
    model.train()
    for i, data in enumerate(train_loader, 0):
        features, labels = data
        features = features.type(torch.FloatTensor)
        features = features.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)

        optimizer.zero_grad()

        # forward
        prediction = model(features).to(device)

        # compute loss and acc
        loss = loss_func(prediction, labels.reshape(-1))
        _, predicted = torch.max(prediction.data, 1)
        acc = (predicted == labels).sum().item() / batchsize

        loss.backward()
        optimizer.step()
        total_loss += loss
        total_acc += acc

        batch_num += 1

    avg_acc = total_acc / batch_num

    print("Training  itr: {:02d} \t total_loss: {:.2f} \t avg_acc: {:.10f} \t".format(itr, total_loss, avg_acc))


    # EVAL
    batch_num_dev, total_acc_dev = 0

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            x_batch, y_batch = data
            x_batch = x_batch.type(torch.FloatTensor)
            x_batch = x_batch.to(device)
            y_batch = y_batch.type(torch.LongTensor)
            y_batch = y_batch.to(device)

            prediction_dev = model(x_batch).to(device)
            _, predicted = torch.max(prediction_dev.data, 1)
            acc_d = (predicted == y_batch).sum().item() / batchsize
            batch_num_dev += 1
            total_acc_dev += acc_d
    avg_acc_dev = total_acc_dev / batch_num_dev
    print("Validation  itr: {:02d} \t avg_acc: {:.10f} \t".format(itr, avg_acc_dev))
    output_model_path = os.path.join(model_foler + 'model_state_' + str(itr) + '.pkl')
    torch.save(model.state_dict(), output_model_path)
    lr_scheduler.step()