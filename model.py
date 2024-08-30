from torch import nn
from torch.nn import Identity
import torch.nn.functional as F


def get_network(model: str, pretrained: bool):
    net = None
    num_features = 0
    match model.lower():
        case 'convnext':
            from torchvision.models import convnext_tiny
            if pretrained:
                from torchvision.models import ConvNeXt_Tiny_Weights
                net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
            else:
                net = convnext_tiny()
            net.classifier[-1] = Identity()
            num_features = 768

        case 'swin':
            from torchvision.models import swin_v2_t
            if pretrained:
                from torchvision.models import Swin_V2_T_Weights
                net = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
            else:
                net = swin_v2_t()
            net.head = Identity()
            num_features = 768

        case 'maxvit':
            from torchvision.models import maxvit_t
            if pretrained:
                from torchvision.models import MaxVit_T_Weights
                net = maxvit_t(weights=MaxVit_T_Weights.DEFAULT)
            else:
                net = maxvit_t()
            net.classifier[-1] = Identity()
            num_features = 512

        case 'vit':
            from torchvision.models import vit_l_16
            if pretrained:
                from torchvision.models import ViT_L_16_Weights
                net = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
            else:
                net = vit_l_16()
            net.heads = Identity()
            num_features = 1024

        case 'vgg16':
            from torchvision.models import vgg16
            if pretrained:
                from torchvision.models import VGG16_Weights
                net = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES)
            else:
                net = vgg16()
            net.classifier = Identity()
            num_features = 4096

    return net, num_features


class TripletModel(nn.Module):
    def __init__(self, model, pretrained=True):
        super(TripletModel, self).__init__()
        net_info = get_network(model, pretrained)
        self.embedding_net = net_info[0]
        self.num_features = net_info[1]

    def forward(self, data):
        res1 = self.embedding_net(data[0])
        res2 = self.embedding_net(data[1])
        res3 = self.embedding_net(data[2])
        res1 = F.normalize(res1)
        res2 = F.normalize(res2)
        res3 = F.normalize(res3)
        return res1, res2, res3

    def get_embedding(self, data):
        res = self.embedding_net(data)
        res = F.normalize(res)
        return res
