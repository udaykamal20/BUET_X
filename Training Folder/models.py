import torch
import torch.nn as nn
import torch.nn.functional as F
from region_loss import RegionLoss
from utils import *

class shift(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(shift, self).__init__()
        self.in_planes = in_planes
        self.kernel_size = kernel_size
        self.channels_per_group = self.in_planes // (self.kernel_size**2)
    
    # Leave the final group in place
    def forward(self,x):

        x_pad = F.pad(x,(1,1,1,1))
        # Alias for convenience
        cpg = self.channels_per_group
        
        cat_layers =[]
        # Bottom shift, grab the Top element
        i = 0
        cat_layers += [x_pad[:, i * cpg : (i + 1) * cpg, :-2, 1:-1]]
        
        # Top shift, grab the Bottom element
        i = 1
        cat_layers += [x_pad[:, i * cpg : (i + 1) * cpg, 2:, 1:-1]]
        
        # Right shift, grab the left element 
        i = 2
        cat_layers += [x_pad[:, i * cpg : (i + 1) * cpg, 1:-1, :-2]]      
        
        # Left shift, grab the right element
        i = 3
        cat_layers += [x_pad[:, i * cpg : (i + 1) * cpg, 1:-1, 2:]]
        
        # Bottom Right shift, grab the Top left element 
        i = 4
        cat_layers += [x_pad[:, i * cpg : (i + 1) * cpg, :-2, :-2]]
        
        # Bottom Left shift, grab the Top right element
        i = 5
        cat_layers += [x_pad[:, i * cpg : (i + 1) * cpg, :-2, 2:]]
        
        # Top Right shift, grab the Bottom Left element
        i = 6
        cat_layers += [x_pad[:, i * cpg : (i + 1) * cpg, 2:, :-2]] 
        
        # Top Left shift, grab the Bottom Right element
        i = 7
        cat_layers += [x_pad[:, i * cpg : (i + 1) * cpg, 2:, 2:]]
        
        # Remaining group left as it is
        i = 8
        cat_layers += [x_pad[:, i * cpg :, 1:-1, 1:-1]]
        return torch.cat(cat_layers,1)

    
class FullChannels_shift_nobias_final(nn.Module):
    def __init__(self):
        super(FullChannels_shift_nobias_final, self).__init__()
        self.width = int(320)
        self.height = int(176)
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0

        def conv_dw_nobias(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.ReLU(inplace=True),
            )
        def conv_dw_nobias_1x1(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.ReLU(inplace=True),
            )

        def conv_shift(inp, oup, stride):
            return nn.Sequential(
                shift(inp),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.ReLU(inplace=True),
            )
        self.model = nn.Sequential(
            conv_dw_nobias_1x1( 3,  32, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw_nobias( 32,  64, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw_nobias( 64, 128, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_shift(128, 256, 1),
            nn.Conv2d(256, 10, 1, 1,bias=False),
        )
        self.loss = RegionLoss([1,1.06357021727,1,2.65376815391],2)

        self.anchors = self.loss.anchors
        self.num_anchors = self.loss.num_anchors
        self.anchor_step = self.loss.anchor_step
    def forward(self, x):
        x = self.model(x)
        return x
    
class FullChannels_shift_all(nn.Module):
    def __init__(self):
        super(FullChannels_shift_all, self).__init__()
        self.width = int(320)
        self.height = int(176)
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0

        def conv_shift(inp, oup, stride):
            return nn.Sequential(
                shift(inp),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.ReLU(inplace=True),
            )
        self.model = nn.Sequential(
            conv_shift( 3,  32, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_shift( 32,  64, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_shift( 64, 128, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_shift(128, 256, 1),
            nn.Conv2d(256, 10, 1, 1,bias=False),
        )
        self.loss = RegionLoss([1,1.06357021727,1,2.65376815391],2)

        self.anchors = self.loss.anchors
        self.num_anchors = self.loss.num_anchors
        self.anchor_step = self.loss.anchor_step
    def forward(self, x):
        x = self.model(x)
        return x
    
