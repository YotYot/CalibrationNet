import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from sem_segmentation.models import ModelBuilder, SegmentationModule
from sem_segmentation.utils import AverageMeter
from sem_segmentation.lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
import sem_segmentation.lib.utils.data as torchdata

class Net(nn.Module):
    def __init__(self,device,  config, num_class=15, mode='classification', channels=64, skip_layer=True):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.batch_norm1 = nn.BatchNorm2d(16, momentum=0.99)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 36, 5,padding=2)
        self.batch_norm2 = nn.BatchNorm2d(36, momentum=0.99)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(36, 64, 5,padding=2)
        self.batch_norm3 = nn.BatchNorm2d(64, momentum=0.99)
        self.pool3 = nn.MaxPool2d(2, 2,ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, 3,padding=1)
        self.batch_norm4 = nn.BatchNorm2d(128, momentum=0.99)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128,256,3, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(256, momentum=0.99)
        self.pool5 = nn.MaxPool2d(2,2)
        self.conv6 = nn.Conv2d(256, num_class, 1)
        self.dense = nn.Linear(512, 256)
        self.dense2 = nn.Linear(256, num_class)
        self.upsampling = nn.ConvTranspose2d(in_channels=num_class, out_channels=num_class,kernel_size=32, stride=32)
        self.conv7_from_pool4 = nn.Conv2d(128, num_class, kernel_size=1)
        self.upsampling16 = nn.ConvTranspose2d(in_channels=num_class, out_channels=num_class, kernel_size=16,stride=16)
        self.conv9_from_pool3 = nn.Conv2d(64, num_class, kernel_size=1)
        self.upsampling8 = nn.ConvTranspose2d(in_channels=num_class, out_channels=num_class, kernel_size=8, stride=8)
        self.conv8_reg = nn.Conv2d(16,1,1,1)
        self.conv8_reg.weight.data[0,:,0,0] = torch.arange(16).float()
        self.conv_from_sem = nn.ConvTranspose2d(in_channels=256, out_channels=16, kernel_size=4, stride=4)
        self.last_conv = nn.Conv2d(32, 16, 1, 1)
        # self.conv8_reg.weight.requires_grad = False
        # self.softmax = nn.Softmax()
        # self.conv8_reg.weight[0,:,0,0] = torch.arange(16).float()
        self.device = device
        self.num_class = num_class
        self.mode = mode
        self.skip_layer = skip_layer
        self.config = config

        self.seg_to_outputs1 = nn.ConvTranspose2d(in_channels=256, out_channels=16, kernel_size=4, stride=4)
        # self.seg_to_outputs2 = nn.ConvTranspose2d(in_channels=150, out_channels=16, kernel_size=8, stride=8)

        builder = ModelBuilder()
        model_path = './sem_segmentation/baseline-resnet50_dilated8-ppm_bilinear_deepsup/'
        suffix = '_epoch_20.pth'
        weights_encoder = os.path.join(model_path, 'encoder' + suffix)
        weights_decoder = os.path.join(model_path, 'decoder' + suffix)
        net_encoder = builder.build_encoder(
            arch='resnet50_dilated8',
            fc_dim=2048,
            weights=weights_encoder)
        net_decoder = builder.build_decoder(
            arch='ppm_bilinear_deepsup',
            fc_dim=2048,
            num_class=150,
            weights=weights_decoder)

        crit = nn.NLLLoss(ignore_index=-1)

        self.segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit, deep_sup_scale=0.4)

        self.segmentation_module = self.segmentation_module.to(device)


    def align_tensor(self, aligned_tensor, input_tensor):
        gapX = input_tensor.shape[2] - aligned_tensor.shape[2]
        gapY = input_tensor.shape[3] - aligned_tensor.shape[3]
        if gapX != 0 and gapY != 0:
            aligned_tensor[:, :, :, :] = input_tensor[:, :, gapX // 2:-gapX // 2, gapY//2:-gapY//2]
        elif gapX !=0:
            aligned_tensor[:, :, :, :] = input_tensor[:, :, gapX // 2:-gapX // 2,:]
        elif gapY != 0:
            aligned_tensor[:, :, :, :] = input_tensor[:, :, :, gapY // 2:-gapY // 2]
        return aligned_tensor

    def forward(self, x):
        feed_dict = dict()
        feed_dict['img_data'] = x
        # semantic_result = self.segmentation_module(feed_dict, segSize=(feed_dict['img_data'].shape[0], feed_dict['img_data'].shape[1]))
        # semantic_result = self.seg_to_outputs1(semantic_result[0]) + self.seg_to_outputs2(semantic_result[1])
        semantic_result = self.segmentation_module.encoder(feed_dict['img_data'], return_feature_maps=True)[0]
        semantic_result = self.conv_from_sem(semantic_result)
        # semantic_result = F.upsample(semantic_result,scale_factor=4)
        # semantic_result = self.seg_to_outputs1(semantic_result)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.pool3(x)
        skip8 = x
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = self.pool4(x)
        skip = x
        x = F.relu(self.batch_norm5(self.conv5(x)))
        x = self.pool5(x)
        x = self.conv6(x)
        if self.mode == 'segmentation':
            # semantic_result = self.segmentation_module.encoder(feed_dict['img_data'])
            # semantic_result = self.segmentation_module(feed_dict)
            x = self.upsampling(x)
            if self.skip_layer:
                skip_upsampled_aligned = torch.zeros_like(x)
                skip_upsampled = self.upsampling16(self.conv7_from_pool4(skip))
                self.align_tensor(skip_upsampled_aligned, skip_upsampled)
                del skip_upsampled
                skip_upsampled8_aligned = torch.zeros_like(x)
                skip_upsampled8 = self.upsampling8(self.conv9_from_pool3(skip8))
                self.align_tensor(skip_upsampled8_aligned, skip_upsampled8)
                del skip_upsampled8
                aligned_semantic = torch.zeros((semantic_result.shape[0], semantic_result.shape[1], x.shape[2], x.shape[3]))
                self.align_tensor(aligned_semantic, semantic_result)
                aligned_semantic = aligned_semantic.to(self.device)
                del semantic_result
                # print (x.shape)
                # print(skip_upsampled_aligned.shape)
                # print(skip_upsampled8_aligned.shape)
                # print (aligned_semantic.shape)
                x = x + skip_upsampled_aligned + skip_upsampled8_aligned
                x = torch.cat((x, aligned_semantic), dim=1)
                x = self.last_conv(x)

            if self.config.target_mode == 'cont':
                x = nn.Softmax()(x)
                x = self.conv8_reg(x)
                x = torch.squeeze(x)
        else:
            x = x.view(-1, self.num_class)
        return x