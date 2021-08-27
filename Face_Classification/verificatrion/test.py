import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd

def test(model, device, dataloader, test_path, test=False):
    label_all = []
    cos_sim_all = []
    model.eval()
    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            output1, output2 = model.forward_both(img1, img2)
            cos_sim = torch.cosine_similarity(output1, output2, dim=1)
            label_all.append(label)
            cos_sim_all.append(np.array(cos_sim.cpu()).ravel())

    array = np.empty(0)
    for i in range(len(cos_sim_all)):
        array = np.append(array, cos_sim_all[i])

    if not test:
        label = np.empty(0)
        for i in range(len(label_all)):
            label = np.append(label, label_all[i])

        y_score = array
        y_true = label
        auc = roc_auc_score(y_true, y_score)
        print("AUC: ", auc)

    else:
        filedata = np.array(pd.read_csv(test_path, header=None))
        dataframe = pd.DataFrame({'Id': filedata.ravel(), 'Category': array})
        dataframe.to_csv("dijingz_s.csv", index=False, sep=',')
    return
