# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from .unet_parts import *

class disparityregression(nn.Module):
    def __init__(self):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(16)),[1,16,1,1])), requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        disp = disp.to(x.device)
        out = torch.sum(x*disp,1)
        return out

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, classification=False, bilinear=False, cont=False):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, bilinear=bilinear)
        self.up2 = up(512, 128, bilinear=bilinear)
        self.up3 = up(256, 64, bilinear=bilinear)
        self.up4 = up(128, 64, bilinear=bilinear)
        self.outc = outconv(64, n_classes)
        self.classification = classification
        self.cont = cont
        self.lin = nn.Linear(512, 16)
        self.regression = disparityregression()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.classification:
            x5 = F.avg_pool2d(x5, 2)
            x5 = self.lin(x5[:,:,0,0])
            return x5
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = F.sigmoid(x)
        if self.cont:
            x = self.regression(x) - 5
        return x
