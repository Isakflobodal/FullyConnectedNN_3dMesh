
import torch.nn as nn
import torch
import torch_geometric.nn as pyg_nn
from torch.nn.functional import conv2d
from createdata import dim, step, BB

PTS = torch.from_numpy(BB).float()

class NeuralNet(nn.Module):
    def __init__(self, layer_sizes, dropout=0.1, device="cuda"):
        super(NeuralNet, self).__init__()
        self.device = device
        layers = []
        input_features = (dim*dim*dim*3 + 24)
        for i, output_features in enumerate(layer_sizes):
            layers.append(nn.Linear(input_features, output_features))
            if i != len(layer_sizes) - 1:
                layers.append(nn.ReLU())
                #layers.append(nn.BatchNorm1d(output_features))
                layers.append(nn.Dropout(dropout))
            input_features = output_features

        self.layers = nn.Sequential(*layers)    

    # forward function: 
    def forward(self, Pc, sdf):                                             # [B,P=8,3] [B,dim,dim,dim] 
        B = Pc.shape[0]
        pts = PTS.to(self.device).unsqueeze(0).expand(B,-1,-1)              # [B,dim*dim*dim,3]

        # includes pts and sdf --> input_features = (dim*dim*dim*4 + 24)
        #sdf = sdf.view(B,-1,1)                                              # [B,dim*dim*dim,1]
        #pts_sdf = torch.cat((pts,sdf), dim = 2)                             # [B,dim*dim*dim,4]
        #pts_sdf = pts_sdf.view(B,-1)                                        # [B,dim*dim*dim*4]

        # includes only pts --> input_features = (dim*dim*dim*3 + 24)
        pts_sdf = pts.view(B,-1)

        Pc = Pc.view(B,-1)                              # [B,8*3=24]
        x = torch.cat((Pc, pts_sdf), dim=1)             # [B,dim*dim*dim*3+24]
        pred = self.layers(x)                           # [B,dim*dim*dim*3]
        pred = pred.view(B,dim,dim,dim,3)               # [B,dim,dim,dim,3]

        return pred