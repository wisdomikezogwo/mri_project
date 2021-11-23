import torch


def get_model(in_channels, out_channels, init_features=8):

    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=in_channels, out_channels=out_channels, init_features=init_features,
                           pretrained=False)
    return model
