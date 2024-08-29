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
            net.classifier = Identity()
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

    return net, num_features


class TripletModel(nn.Module):
    def __init__(self, model, pretrained=True):
        super(TripletModel, self).__init__()
        net_info = get_network(model, pretrained)
        self.embedding_net = net_info[0]
        self.num_features = net_info[1]

    def forward(self, data):
        res1 = self.embedding_net(data[0]).view(-1, self.num_features)
        res2 = self.embedding_net(data[1]).view(-1, self.num_features)
        res3 = self.embedding_net(data[2]).view(-1, self.num_features)
        res1 = F.normalize(res1)
        res2 = F.normalize(res2)
        res3 = F.normalize(res3)
        return res1, res2, res3

    def get_embedding(self, data):
        res = self.embedding_net(data).view(-1, self.num_features)
        res = F.normalize(res)
        return res
