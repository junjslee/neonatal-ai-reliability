import torch.nn as nn
import torch

class ClassificationLoss(nn.Module):
    # Modify init to accept pre-calculated pos_weight
    def __init__(self, classification_weight=1.0, pos_weight_tensor: torch.Tensor = None):
        super().__init__()
        self.classification_weight = classification_weight
        # Pass the pos_weight tensor during initialization
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    def forward(self, cls_pred: torch.Tensor, cls_gt: torch.Tensor) -> tuple[torch.Tensor, dict]: # Return tuple
        # Ensure cls_gt is Float (BCEWithLogitsLoss expects float targets)
        loss = self.bce_loss(cls_pred, cls_gt.float()) * self.classification_weight
        # Ensure loss_dict keys match what might be expected downstream
        loss_dict = {'loss_cls': loss.item()} # Changed key slightly for potential clarity
        return loss, loss_dict
