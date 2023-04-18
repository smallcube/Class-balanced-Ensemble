import torch

def acc(output, labels):
    _, preds = torch.max(output, 1)
   
    acc = torch.sum(preds == labels.data).float()/output.shape[0]

    return acc
