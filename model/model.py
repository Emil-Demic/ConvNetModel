import torch
import torch.nn as nn
from .sa import Self_Attention
from .ca import Cross_Attention
from .rn import Relation_Network, cos_similar

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args

        self.sa = Self_Attention(d_model=args.d_model, cls_number=args.cls_number, pretrained=args.pretrained)
        self.ca = Cross_Attention(args=args, h=args.head, n=args.number, d_model=args.d_model, d_ff=args.d_ff, dropout=0.1)
        self.conv2d = nn.Conv2d(768, 512, 2, 2)


    def forward(self, sk, im, stage='train', only_sa=False):

        if stage == 'train':

            sk_im = torch.cat((sk, im), dim=0)
            sa_fea, left_tokens, idxs = self.sa(sk_im)  # [4b, 197, 768]
            ca_fea = self.ca(sa_fea)  # [4b, 197, 768]

            cls_fea = ca_fea[:, 0]  # [4b, 1, 768]
            return cls_fea

        else:

            if only_sa:
                sa_fea, left_tokens, idxs = self.sa(sk)  # [b, 197, 768]
                return sa_fea, idxs
            else:
                sk_im = torch.cat((sk, im), dim=0)
                ca_fea = self.ca(sk_im)  # [2b, 197, 768]

                cls_fea = ca_fea[:, 0]  # [2b, 1, 768]

                # print('cls_fea:', cls_fea.size())
                # print('rn_scores:', cls_fea.size())
                return cls_fea
