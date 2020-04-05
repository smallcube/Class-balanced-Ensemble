import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=4.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target, sample_weight, weight=None, n=1.0, gamma=2.0):
        """
        input:[n, c]
        target:[n,]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        index = target.view(len(input), 1).long()

        p = pt.gather(1, index)
        p = torch.cat((sample_weight, p), 1)
        #maxP, maxIndex = torch.max(p, 1)
        maxP = torch.mean(p, 1)
        maxP = maxP.view(len(input), 1)
        
        logpt = (1-maxP)**gamma*logpt
        #logpt = (1-pt)*logpt
        loss = F.nll_loss(logpt, target, weight)
        return loss, p