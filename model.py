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

        case 'resnext':
            from torchvision.models import resnext50_32x4d
            if pretrained:
                from torchvision.models import ResNeXt50_32X4D_Weights
                net = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
            else:
                net = resnext50_32x4d()
            net.fc = Identity()
            num_features = 2048

        case 'resnet':
            from torchvision.models import resnet152
            if pretrained:
                from torchvision.models import ResNet152_Weights
                net = resnet152(weights=ResNet152_Weights.DEFAULT)
            else:
                net = resnet152()
            net.fc = Identity()
            num_features = 2048

        case 'regnet_x':
            from torchvision.models import regnet_x_3_2gf
            if pretrained:
                from torchvision.models import RegNet_X_3_2GF_Weights
                net = regnet_x_3_2gf(weights=RegNet_X_3_2GF_Weights.DEFAULT)
            else:
                net = regnet_x_3_2gf()
            net.fc = Identity()
            num_features = 1008

        case 'regnet_y':
            from torchvision.models import regnet_y_3_2gf
            if pretrained:
                from torchvision.models import RegNet_Y_3_2GF_Weights
                net = regnet_y_3_2gf(weights=RegNet_Y_3_2GF_Weights.DEFAULT)
            else:
                net = regnet_y_3_2gf()
            net.fc = Identity()
            num_features = 1512

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
            from torchvision.models import vit_b_32
            if pretrained:
                from torchvision.models import ViT_B_32_Weights
                net = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
            else:
                net = vit_b_32()
            net.heads = Identity()
            num_features = 768

        case 'efficientnet':
            from torchvision.models import efficientnet_b4
            if pretrained:
                from torchvision.models import EfficientNet_B4_Weights
                net = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
            else:
                net = efficientnet_b4()
            net.classifier = Identity()
            num_features = 1792

        case 'shufflenet':
            from torchvision.models import shufflenet_v2_x2_0
            if pretrained:
                from torchvision.models import ShuffleNet_V2_X2_0_Weights
                net = shufflenet_v2_x2_0(weights=ShuffleNet_V2_X2_0_Weights.DEFAULT)
            else:
                net = shufflenet_v2_x2_0()
            net.fc = Identity()
            num_features = 2048

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
