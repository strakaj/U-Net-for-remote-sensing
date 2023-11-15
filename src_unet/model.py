import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class SMP_Base_Unet(nn.Module):
    """ 
    Pytorch segmentation U-Net with ResNet34 (default) 
    with added metadata information at encoder output
    
    """

    def __init__(self,
                 n_channels,
                 n_classes,
                 encoder_name
                 ):
        super(SMP_Base_Unet, self).__init__()

        self.reduction_conv = None
        if "mit" in encoder_name:
            self.reduction_conv = nn.Conv2d(n_channels, 3, 1)
            n_channels = 3

        self.seg_model = smp.create_model(arch="unet", encoder_name=encoder_name, classes=n_classes,
                                          in_channels=n_channels)

    def forward(self, x):

        if self.reduction_conv is not None:
            x = self.reduction_conv(x)
        output = self.seg_model(x)

        return output
