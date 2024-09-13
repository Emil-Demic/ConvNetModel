from torch import nn
from torch.nn import Identity, Linear, AdaptiveAvgPool2d, AdaptiveMaxPool2d
import torch.nn.functional as F


def get_network(model: str, pretrained: bool):
    net = None
    num_features = 0
    match model.lower():
        case 'convnext':
            from torchvision.models import convnext_base
            if pretrained:
                from torchvision.models import ConvNeXt_Base_Weights
                net = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
            else:
                net = convnext_base()
            net.classifier[-1] = Identity()
            num_features = 1024

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
            from torchvision.models import vit_b_16
            if pretrained:
                from torchvision.models import ViT_B_16_Weights
                net = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            else:
                net = vit_b_16()
            net.heads = Identity()
            num_features = 768

        case 'vgg16':
            from torchvision.models import vgg16
            if pretrained:
                from torchvision.models import VGG16_Weights
                net = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features
            else:
                net = vgg16().features
            num_features = 512

    return net, num_features


class TripletModel(nn.Module):
    def __init__(self, model, pretrained=True):
        super(TripletModel, self).__init__()
        net_info = get_network(model, pretrained)
        self.embedding_net = net_info[0]
        self.num_features = net_info[1]
        self.pool = AdaptiveAvgPool2d(1)

    def forward(self, data):
        res1 = self.embedding_net(data[0])
        res2 = self.embedding_net(data[1])
        res3 = self.embedding_net(data[2])
        # res1 = self.pool(res1).view(-1, self.num_features)
        # res2 = self.pool(res2).view(-1, self.num_features)
        # res3 = self.pool(res3).view(-1, self.num_features)
        res1 = F.normalize(res1)
        res2 = F.normalize(res2)
        res3 = F.normalize(res3)
        return res1, res2, res3

    def get_embedding(self, data):
        res = self.embedding_net(data)
        # res = self.pool(res).view(-1, self.num_features)
        res = F.normalize(res)
        return res
