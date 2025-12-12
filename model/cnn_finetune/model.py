import segmentation_models_pytorch as smp
import torch.nn as nn


class LateralClassificationModel(nn.Module):
    def __init__(self, layers, aux_params=None):
        super().__init__()
        # For lateral images, we use 1 input channel (or 3 for mit encoders)
        self.is_mit_encoder = 'mit' in layers
        in_channels = 3 if self.is_mit_encoder else 1
        # Create a U-Net from SMP; we only need the encoder part.
        self.base_model = smp.Unet(layers, encoder_weights='imagenet', in_channels=in_channels, classes=1, aux_params=aux_params)
        
    def forward(self, x):
        # For non-mit encoders with single channel images, replicate to 3 channels.
        if self.is_mit_encoder and x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.base_model.encoder(x)
        bottleneck = features[-1]
        
        cls_logits = self.base_model.classification_head(bottleneck)
        return cls_logits
        

