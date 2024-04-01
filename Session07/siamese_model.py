import torch
import torch.nn as nn
from torchvision import models

class NormLayer(nn.Module):
    """ Layer that computer embedding normalization """
    def __init__(self, l=2):
        """ Layer initializer """
        assert l in [1, 2]
        super().__init__()
        self.l = l
        return
    
    def forward(self, x):
        """ Normalizing embeddings x. The shape of x is (B,D) """
        x_normalized = x / torch.norm(x, p=self.l, dim=-1, keepdim=True)
        return x_normalized

def get_children(model: torch.nn.Module):
    # function used to properly decompose resnet
    decomposed = []
    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            for block in layer:
                decomposed.append(block)
        else:
            decomposed.append(layer)

    return decomposed

class TriNet_Siamese_model(nn.Module):

    def __init__(self, emb_dim, pretrained = False) -> None:
        super().__init__()

        # set weights for resnet18
        if pretrained == False:
            weights = "DEFAULT"
        elif pretrained == True:
            weights = "IMAGENET1K_V1"

        resnet18 = models.resnet18(weights=weights)

        modules = get_children(resnet18)
        # discard last two layers: avgpool and linear
        modules = modules[:-2]

        self.resnet18 = nn.Sequential(*modules)
        # hard coded dims
        self.fc = nn.Linear(in_features=512*8*8, out_features=emb_dim)

        self.flatten = nn.Flatten()
        self.norm = NormLayer()

    def forward(self, anchor, positive, negative):
        """ Forwarding a triplet """
        imgs = torch.concat((anchor, positive, negative), dim=0)
        randoms = torch.zeros(([1, 3, 224, 224])).to("cuda")
        # forward pass
        x = self.resnet18(imgs)
        x_flat = self.flatten(x)
        x_emb = self.fc(x_flat)
        x_emb_norm = self.norm(x_emb)
        # decompose batch
        anchor_emb, positive_emb, negative_emb = torch.chunk(x_emb_norm, 3, dim=0)
        
        return anchor_emb, positive_emb, negative_emb