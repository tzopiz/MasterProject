import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    Computes the Dice Similarity Coefficient between prediction and target.
    """
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        
        intersection = (pred * target).sum(dim=(2, 3))
        
        loss = (1 - ((2. * intersection + self.smooth) / 
                     (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth)))
        
        return loss.mean()

