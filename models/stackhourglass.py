from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from submodule import *
from models.dfd import Dfd_net

class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post

class PSMNet(nn.Module):
    def  __init__(self, maxdisp, dfd_net = True, dfd_at_end=True, device=None,right_head=False, pred_oof=False, mono_pred=False):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.dfd_net = dfd_net
        self.dfd_at_end = dfd_at_end
        self.device = device
        self.right_head = right_head
        self.pred_oof = pred_oof
        self.mono_pred = mono_pred

        if self.dfd_net and not self.dfd_at_end:
            features = 66
        else:
            features = 64

        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(convbn_3d(features, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        if self.right_head:

            self.dres5 = hourglass(32)

            self.dres6 = hourglass(32)

            self.dres7 = hourglass(32)

        if self.pred_oof:

            self.dres8 = hourglass(32)

            self.dres9 = hourglass(32)

            self.dres10 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        if self.right_head:
            self.classif4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

            self.classif5 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

            self.classif6 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        if self.pred_oof:
            self.classif7 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

            self.classif8 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

            self.classif9 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.last_conv = nn.Conv2d(2,1,1)
        # self.conv1 = nn.Conv2d(192,32,1)
        # self.conv2 = nn.Conv2d(32,1,1)
        # self.conv3 = nn.Conv2d(16,1,1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        # self.last_conv.weight.data = torch.Tensor([[[[0.5]],[[0.5]]]])

        if self.mono_pred:
            self.dfd = Dfd_net(mode='segmentation', target_mode='cont', pool=False)
            self.dfd.to(device)
            # model_path = 'models/dfd_checkpoint/checkpoint_274.pth.tar'
            # print("loading checkpoint for dfd net")
            # checkpoint = torch.load(model_path, map_location=device)
            # self.dfd.load_state_dict(checkpoint['state_dict'], strict=True)
            # #Freeze the net if at features level
            # if not self.dfd_at_end:
            #     print("Freezing dfd net")
            #     for child in self.dfd.children():
            #         for param in child.parameters():
            #             param.requires_grad = False


    def forward(self, left, right):

        refimg_fea     = self.feature_extraction(left)
        targetimg_fea  = self.feature_extraction(right)

        if self.mono_pred:
            mono_pred, mono_conf = self.dfd(left)

        #matching
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()

        for i in range(self.maxdisp//4):
            if i > 0 :
             cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
             cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
            else:
             cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
             cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None) 
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1) 
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2) 
        out3 = out3+cost0

        if self.right_head:
            out4, pre4, post4 = self.dres5(cost0, None, None)
            out4 = out4 + cost0

            out5, pre5, post5 = self.dres6(out4, pre4, post4)
            out5 = out5 + cost0

            out6, pre6, post6 = self.dres7(out5, pre4, post5)
            out6 = out6 + cost0

        if self.pred_oof:
            out7, pre7, post7 = self.dres8(cost0, None, None)
            out7 = out7 + cost0

            out8, pre8, post8 = self.dres9(out7, pre7, post7)
            out8 = out8 + cost0

            out9, pre9, post9 = self.dres10(out8, pre8, post8)
            out9 = out9 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.right_head:
            cost4 = self.classif4(out4)
            cost5 = self.classif5(out5) + cost4
            cost6 = self.classif6(out6) + cost5

        if self.pred_oof:
            cost7 = self.classif4(out7)
            cost8 = self.classif5(out8) + cost7
            cost9 = self.classif6(out9) + cost8

        if self.training:
            cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear', align_corners=True)
            cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear', align_corners=True)

            cost1 = torch.squeeze(cost1,1)
            pred1 = F.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2,1)
            pred2 = F.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)

            if self.right_head:
                cost4 = F.upsample(cost4, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear',
                                   align_corners=True)
                cost5 = F.upsample(cost5, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear',
                                   align_corners=True)

                cost4 = torch.squeeze(cost4, 1)
                pred4 = F.softmax(cost4, dim=1)
                pred4 = disparityregression(self.maxdisp)(pred4)

                cost5 = torch.squeeze(cost5, 1)
                pred5 = F.softmax(cost5, dim=1)
                pred5 = disparityregression(self.maxdisp)(pred5)

            if self.pred_oof:
                cost7 = F.upsample(cost7, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear',
                                   align_corners=True)
                cost8 = F.upsample(cost8, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear',
                                   align_corners=True)

                cost7 = torch.squeeze(cost7, 1)
                pred7 = F.softmax(cost7, dim=1)
                pred7 = disparityregression(self.maxdisp)(pred7)

                cost8 = torch.squeeze(cost8, 1)
                pred8 = F.softmax(cost8, dim=1)
                pred8 = disparityregression(self.maxdisp)(pred8)


        cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear', align_corners=True)
        cost3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(cost3,dim=1)
        conf192 = pred3
        pred3 = disparityregression(self.maxdisp)(pred3)

        if self.right_head:
            cost6 = F.upsample(cost6, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear',
                               align_corners=True)
            cost6 = torch.squeeze(cost6, 1)
            pred6 = F.softmax(cost6, dim=1)
            pred6 = disparityregression(self.maxdisp)(pred6)

        if self.pred_oof:
            cost9 = F.upsample(cost9, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear',
                               align_corners=True)
            cost9 = torch.squeeze(cost9, 1)
            pred9 = F.softmax(cost9, dim=1)
            pred9 = disparityregression(self.maxdisp)(pred9)

        if self.mono_pred:
            stereo_conf = ones_vector(192)(conf192)
            stereo_conf = torch.unsqueeze(stereo_conf,1)
            mono_conf = ones_vector(16)(mono_conf)
            mono_conf = torch.unsqueeze(mono_conf, 1)
            fuse_conf = torch.cat((mono_conf, stereo_conf), 1)
            fuse_conf = nn.Softmax(dim=1)(fuse_conf)
            fuse = fuse_conf[:, 0, :, :] * mono_pred + fuse_conf[:, 1, :, :] * pred3

        if self.training:
            if self.pred_oof:
                return pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8, pred9
            elif self.right_head:
                return pred1, pred2, pred3, pred4, pred5, pred6
            elif self.mono_pred:
                return pred1, pred2, pred3, mono_pred, fuse, conf192, mono_conf
            else:
                return pred1, pred2, pred3
        elif self.pred_oof:
            return pred3, pred6, pred9
        elif self.right_head:
            return pred3, pred6, conf192
        elif self.mono_pred:
            return pred3, mono_pred, fuse
        else:
            return conf192, pred3
