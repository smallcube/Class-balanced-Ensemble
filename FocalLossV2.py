import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target, sample_weight, weight=None):
        _, preds = torch.max(input, 1)
        acc = torch.sum((preds==target.data).float()*sample_weight).float()/torch.sum(sample_weight)

        sample_weight1 = sample_weight.clone().detach()
        
        logpt = F.log_softmax(input, dim=1)
        #logpt = logpt * sample_weight1
        #loss = F.nll_loss(logpt, target, class_weight)
        loss = F.nll_loss(logpt, target, weight, reduction='none')
        loss1 = torch.mean(loss*sample_weight1)

        #loss1 = loss
        #loss1 = loss
        w1 = acc

        if acc>1.0 and acc<1.0:
            reweight = acc/(1.0-acc)
            w1 = math.log(reweight)

            mask = (preds==target)
            #sample_weight[mask] = sample_weight[mask]*math.exp(-w1)
            sample_weight[~mask] = sample_weight[~mask]*reweight
            sample_weight = sample_weight/torch.sum(sample_weight)
            sample_weight = sample_weight.clone().detach()
        #loss1 = torch.mean(loss1)
        #print(loss1)
        return loss1, sample_weight, w1