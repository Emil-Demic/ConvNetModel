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
        res1 = F.normalize(self.embedding_net(data[0]))
        res2 = F.normalize(self.embedding_net(data[1]))
        res3 = F.normalize(self.embedding_net(data[2]))
        return res1.view(-1, self.num_features), res2.view(-1, self.num_features), res3.view(-1, self.num_features)

    def get_embedding(self, data):
        res = F.normalize(self.embedding_net(data))
        return res.view(-1, self.num_features)
