from torch import nn
from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d, Identity
import torch.nn.functional as F


def get_network(model: str, pretrained: bool):
    net = None
    num_features = 0
    match model.lower():
        case 'convnext':
            from torchvision.models import convnext_small
            if pretrained:
                from torchvision.models import ConvNeXt_Small_Weights
                net = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT).features
            else:
                net = convnext_small().features

            net[7] = Identity()
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


class SbirModel(nn.Module):
    def __init__(self, model, pretrained=True):
        super(SbirModel, self).__init__()
        net_info = get_network(model, pretrained)
        self.embedding_net = net_info[0]
        self.num_features = net_info[1]
        if model == 'vgg16':
            self.pool = AdaptiveMaxPool2d(1)
        else:
            self.pool = AdaptiveAvgPool2d(1)

    def forward(self, data):
        res1 = self.embedding_net(data[0])
        res2 = self.embedding_net(data[1])
        res1 = self.pool(res1).view(-1, self.num_features)
        res2 = self.pool(res2).view(-1, self.num_features)
        res1 = F.normalize(res1)
        res2 = F.normalize(res2)
        return res1, res2

    def get_embedding(self, data):
        res = self.embedding_net(data)
        res = self.pool(res).view(-1, self.num_features)
        res = F.normalize(res)
        return res
