import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image
import os

from models.dfd import Dfd_net,psi_to_depth
from models.stackhourglass import PSMNet
from models.MiDaS import MonoDepthNet
from unet.predict_cont import predict_full_img, get_Unet
from configurable_stn_projective import ConfigNet
from ImageLoader import myImageloader

from local_utils import depth2disparity

# from PyQt4 import QtGui
# from PyQt4.QtGui import *
from PyQt5 import QtGui
from PyQt5.QtGui import *

class CalibTrain():
    def __init__(self, mono_net, lr, scheduled_lr=False):
        self.mono_net = mono_net
        self.device = torch.device('cuda:0')
        self.stereo_model = self.get_stereo_model()
        self.mono_model = self.get_mono_model(mono_net)
        self.stereo_model_for_calibration = self.get_stereo_model()
        self.lr = lr
        self.calibration_model = self.get_calibration_model()
        self.optimizer = self.get_optimizer(self.calibration_model)
        self.dir_checkpoint = 'checkpoints/demo_cp'
        self.best_test_loss = 20.0
        self.im_cnt = 0
        self.patch_size = 256
        self.scheduled_lr = scheduled_lr

    def get_stereo_model(self):
        stereo_model = PSMNet(192, device=self.device, dfd_net=False, dfd_at_end=False, right_head=False)
        stereo_model = nn.DataParallel(stereo_model)
        stereo_model.cuda()
        state_dict = torch.load('checkpoints/PSM/pretrained_model_KITTI2015.tar')
        stereo_model.load_state_dict(state_dict['state_dict'], strict=False)
        stereo_model.train()
        return stereo_model

    def get_mono_model(self, mono_net):
        if mono_net == 'phase-mask':
            mono_model = Dfd_net(mode='segmentation', target_mode='cont', pool=False)
            mono_model = mono_model.eval()
            mono_model = mono_model.to(self.device)
            model_path='checkpoints/Dfd/checkpoint_257.pth.tar'
            print("loading checkpoint from: ", model_path)
            checkpoint = torch.load(model_path, map_location=self.device)
            mono_model.load_state_dict(checkpoint['state_dict'], strict=False)

        elif mono_net == 'midas':
            # load network
            midas_model_path = 'checkpoints/Midas/model.pt'
            mono_model = MonoDepthNet(midas_model_path)
            mono_model.to(self.device)
            mono_model.eval()
        else:
            mono_model = get_Unet('models/unet/CP100_w_noise.pth', device=self.device)
        return mono_model

    def get_calibration_model(self):
        model =  ConfigNet(stereo_model=self.stereo_model_for_calibration, stn_mode='projective', ext_disp2depth=False, device=self.device).to(self.device)
        for param in model.stereo_model.parameters():
            param.requires_grad = False
        return model

    def get_optimizer(self, model, epoch=None):
        global loss_to_compare, loss_list
        if epoch is not None:
            if epoch == 1:
                loss_to_compare = loss_list[0]
                print ("First Loss: {}".format(loss_to_compare))
            elif epoch > 1:
                if loss_list[-1] < loss_to_compare * 0.9:
                    loss_to_compare = loss_list[-1]
                    self.lr = self.lr / 2
                    print ("Updating LR to {}".format(self.lr))
            optimizer = optim.SGD(model.parameters(), lr=self.lr)
            # lr = self.lr / ((epoch+1)**0.5)
            # optimizer = optim.SGD(model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=self.lr)
        return optimizer

    def get_small_mono(self, mono_out):
        mono_out_normalized = (mono_out - torch.min(mono_out)) / (torch.max(mono_out) / torch.min(mono_out))
        mono_out_small = transforms.ToPILImage()(mono_out_normalized[0].cpu())
        mono_out_small = mono_out_small.resize((320, 256), resample=Image.LANCZOS)
        mono_out_small = transforms.ToTensor()(mono_out_small).to(self.device)
        mono_out = ((mono_out_small) * (torch.max(mono_out) / torch.min(mono_out)) + torch.min(mono_out))
        return mono_out

    def get_mask(self, stereo_out, right_transformed):
        mono_mask = (stereo_out > 0.5) & (stereo_out < 4.5)
        mask = (right_transformed != 0)[:, 0, :, :]
        mask = mask & mono_mask
        nan_mask = (~(torch.isnan(right_transformed)))[:, 0, :, :]
        mask = mask & nan_mask
        return mask

    def show_depth_maps(self, left, right_transformed, mono, stereo_rect, stereo, blocking=False):
        fig, ax_list, loss_list
        vmin = torch.min(stereo_rect).item()
        vmax = torch.max(stereo_rect).item()
        # plt.figure(figsize=(18,6))
        if self.im_cnt == 0:
            ax_list[0].imshow(left[0].permute(1, 2, 0).detach().cpu())
            plt.setp(ax_list[0].get_xticklabels(), visible=False)
            plt.setp(ax_list[0].get_yticklabels(), visible=False)
            ax_list[0].tick_params(axis='both', which='both', length=0)
            ax_list[0].set_title('Left Image')

            ax_list[1].imshow(right_transformed[0].permute(1, 2, 0).detach().cpu(), cmap='jet', vmin=vmin, vmax=vmax)
            plt.setp(ax_list[1].get_xticklabels(), visible=False)
            plt.setp(ax_list[1].get_yticklabels(), visible=False)
            ax_list[1].tick_params(axis='both', which='both', length=0)
            ax_list[1].set_title('Right (transformed) Image')

            ax_list[2].imshow(mono[0].detach().cpu(), cmap='jet', vmin=vmin, vmax=vmax)
            plt.setp(ax_list[2].get_xticklabels(), visible=False)
            plt.setp(ax_list[2].get_yticklabels(), visible=False)
            ax_list[2].tick_params(axis='both', which='both', length=0)
            ax_list[2].set_title('Monocular Depth Map')

            ax_list[3].imshow(stereo_rect[0].detach().cpu(), cmap='jet', vmin=vmin, vmax=vmax)
            plt.setp(ax_list[3].get_xticklabels(), visible=False)
            plt.setp(ax_list[3].get_yticklabels(), visible=False)
            ax_list[3].tick_params(axis='both', which='both', length=0)
            ax_list[3].set_title('Stereo Before Calibration')

            ax_list[4].imshow(stereo[0].detach().cpu(), cmap='jet', vmin=vmin, vmax=vmax)
            plt.setp(ax_list[4].get_xticklabels(), visible=False)
            plt.setp(ax_list[4].get_yticklabels(), visible=False)
            ax_list[4].tick_params(axis='both', which='both', length=0)
            ax_list[4].set_title('Stereo After Calibration')

            ax_list[5].plot(loss_list, 'b')
            ax_list[5].set_title('Train Loss (Mono vs. Stereo)')
            # plt.setp(ax_list[5].get_xticklabels(), visible=False)
            # plt.setp(ax_list[5].get_yticklabels(), visible=False)
            # ax_list[5].tick_params(axis='both', which='both', length=0)
        else:
            # plt.subplot(155)
            # plt.xticks(visible=False)
            # plt.yticks(visible=False)
            ax_list[4].imshow(stereo[0].detach().cpu(), cmap='jet', vmin=vmin, vmax=vmax)
            ax_list[1].imshow(right_transformed[0].permute(1, 2, 0).detach().cpu())
            ax_list[5].plot(loss_list, 'b')
        if blocking:
            fig.canvas.draw()
            plt.ioff()
            plt.show()
        else:
            fig.canvas.draw()
        # plt.savefig('figures/tmp/dfd_'+str(im_cnt)+'.png')
        # plt.close()
        plt.pause(0.0001)
        self.im_cnt += 1
        # plt.show()

    def show_best_calibration(self, obj, cp_file,  small_left, small_right, left, stereo_unrect, mono_out):
        self.calibration_model.train()
        state_dict = torch.load(cp_file)
        self.calibration_model.load_state_dict(state_dict)
        with torch.no_grad():
            stereo_out, theta, right_transformed = self.calibration_model(small_left, small_right)
        # stereo_out = 100 / stereo_out
        if self.mono_net == 'midas':
            mono_out_for_train = mono_out
            # mono_out_for_train = mono_out * (torch.mean(stereo_out[stereo_out < 3.0]) / torch.mean(mono_out[mono_out < 3.0]))
        else:
            mono_out_for_train = mono_out
        stereo_out = stereo_out - torch.mean(stereo_out)
        stereo_out = stereo_out / (torch.max(stereo_out) - torch.min(stereo_out))
        stereo_out = stereo_out - torch.min(stereo_out)
        stereo_out = stereo_out * 100
        obj.right.show_right_transformed(right_transformed.cpu().detach().numpy())
        obj.calib_depth.handle(stereo_out.cpu().detach().numpy())
        # self.show_depth_maps(left, right_transformed, mono_out_for_train, stereo_unrect, stereo_out, blocking=True)


    def get_mono_and_unrect_stereo(self, left, small_left, small_right):
        self.stereo_model.eval()
        with torch.no_grad():
            if self.mono_net == 'midas':
                mono_out = self.mono_model.forward(small_left)
                mono_out = torch.squeeze(mono_out, 0)
                mono_out -= torch.mean(mono_out)
                mono_out /= (torch.max(mono_out) - torch.min(mono_out))
                mono_out -= torch.min(mono_out)
                mono_out *= 100
            elif self.mono_net == 'phase-mask':
                dfd_mono_out, _ = self.mono_model(left, focal_point=1.5)
                mono_out = torch.unsqueeze(dfd_mono_out, 0)
                mono_out = torch.squeeze(depth2disparity(torch.unsqueeze(self.get_small_mono(mono_out),0), mono_out.device), 0)
            else:
                mono_out_unet = predict_full_img(self.mono_model, left, device=self.device)
                mono_out_unet = psi_to_depth(mono_out_unet, focal_point=1.5)
                mono_out = torch.unsqueeze(mono_out_unet, 0)

            ##Make mean = 0, and values in the range 0-1

            _, stereo_unrect = self.stereo_model(small_left, small_right)
            stereo_unrect -= torch.mean(stereo_unrect)
            stereo_unrect -= torch.min(stereo_unrect)
            stereo_unrect /= (torch.max(stereo_unrect) - torch.min(stereo_unrect))
            stereo_unrect *= 100
        # if self.mono_net == 'midas':
        #     mono_out -= torch.min(mono_out)
        #     # mono_out = torch.clamp(1 / mono_out, 0, 3)
        # else:
        #     mono_out = self.get_small_mono(mono_out)
        return stereo_unrect, mono_out

    def train(self, small_left, small_right, mono_out):
        global fig, ax_list, loss_list
        self.calibration_model.train()
        self.stereo_model.eval()
        self.optimizer.zero_grad()
        stereo_out, theta, right_transformed = self.calibration_model(small_left, small_right)
        # stereo_out = 100 / stereo_out

        if self.mono_net == 'midas':
            stereo_out = stereo_out - torch.mean(stereo_out)
            stereo_out = stereo_out / (torch.max(stereo_out) - torch.min(stereo_out))
            stereo_out = stereo_out - torch.min(mono_out)
            stereo_out = stereo_out * 100
            loss = F.l1_loss(stereo_out, mono_out)
        else:
            loss = F.l1_loss(stereo_out, mono_out)
            stereo_out = stereo_out - torch.mean(stereo_out)
            stereo_out = stereo_out / (torch.max(stereo_out) - torch.min(stereo_out))
            stereo_out = stereo_out - torch.min(mono_out)
            stereo_out = stereo_out * 100

        # stereo_out = torch.clamp(stereo_out, 0.05, 0.95)

        # if self.mono_net == 'midas':
        #     mono_out_for_train = mono_out
        #     # mono_out_for_train = mono_out * (torch.mean(stereo_out[stereo_out < 3.0]) / torch.mean(mono_out[mono_out < 3.0]))
        # else:
        #     mono_out_for_train = mono_out

        # mask = self.get_mask(stereo_out, right_transformed)
        loss_list.append(loss)

        # self.show_depth_maps(left, right_transformed, mono_out_for_train, stereo_unrect, stereo_out)

        loss.backward()
        self.optimizer.step()

        return loss, right_transformed, stereo_out

    def get_mono_prediction(self, left_img, right_img):
        # reshape the image data as 1dimensional array
        left_train_filelist = [left_img]
        right_train_filelist = [right_img]
        train_db = myImageloader(left_img_files=left_train_filelist, right_img_files=right_train_filelist,
                                 supervised=False,
                                 train_patch_w=self.patch_size,
                                 transform=transforms.Compose(
                                     [transforms.ToTensor()]),
                                 label_transform=transforms.Compose([transforms.ToTensor()]),
                                 get_filelist=(left_img is None))

        train_loader = torch.utils.data.DataLoader(train_db, batch_size=1, shuffle=True, num_workers=0)
        for batch_idx, (left, right, small_left, small_right) in enumerate(train_loader):
            left, right, small_left, small_right = left.to(self.device), right.to(self.device), small_left.to(
                self.device), small_right.to(self.device)

        stereo_unrect, mono_out = self.get_mono_and_unrect_stereo(left, small_left, small_right)

        return stereo_unrect, mono_out

    def get_stereo_prediction(self, image_data):
        image = image_data.as_1d_image()
        return QtGui.QImage(image.data, image_data.mem_info.width, image_data.mem_info.height,
                            QtGui.QImage.Format_RGB888)

    def sample_images(self, l_img, r_img):
        if l_img is not None:
            right_train_filelist = [r_img]
            left_train_filelist = [l_img]
        else:
            right_train_filelist = ['Sample_Images/R_10.tif']
            left_train_filelist = ['Sample_Images/L_10.tif']

        train_db = myImageloader(left_img_files=left_train_filelist, right_img_files=right_train_filelist,
                                 supervised=False,
                                 train_patch_w=self.patch_size,
                                 transform=transforms.Compose(
                                     [transforms.ToTensor()]),
                                 label_transform=transforms.Compose([transforms.ToTensor()]), get_filelist=(l_img is None))

        train_loader = torch.utils.data.DataLoader(train_db, batch_size=1, shuffle=True, num_workers=0)
        for batch_idx, (left, right, small_left, small_right) in enumerate(train_loader):
            left, right, small_left, small_right = left.to(self.device), right.to(self.device), small_left.to(
                self.device), small_right.to(self.device)
        self.left = left
        self.right = right
        self.small_left = small_left
        self.small_right = small_right


    def calibrate(self, obj, l_img=None, r_img=None, epoch_num = 10):
        self.sample_images(l_img, r_img)
        global fig, ax_list, loss_list
        loss_list = list()
        stereo_unrect, mono_out = self.get_mono_and_unrect_stereo(self.left, self.small_left, self.small_right)

        ax = obj.figure.add_subplot(111)
        ax.clear()
        for param_group in self.optimizer.param_groups:
            print ("Training with LR: {}".format(param_group['lr']))

        for e in range(epoch_num):
            if self.scheduled_lr:
                self.optimizer = self.get_optimizer(model=self.calibration_model, epoch=e)
                for param_group in self.optimizer.param_groups:
                    print("Trainin with LR: {}".format(param_group['lr']))
            loss, right_transformed, stereo_out = self.train(self.small_left, self.small_right, mono_out)
            ax.plot(loss_list, '*-', color='g')
            obj.loss.draw()
            # obj.rightTransformed.handle(right_transformed.cpu().detach().numpy())
            obj.right.show_right_transformed(right_transformed.cpu().detach().numpy())
            obj.calib_depth.handle(stereo_out.cpu().detach().numpy())
            if loss < self.best_test_loss:
                cp_file = os.path.join(self.dir_checkpoint, 'CP{}.pth'.format(e))
                torch.save(self.calibration_model.state_dict(),
                           cp_file)
                print('Checkpoint {} saved !'.format(e))
                self.best_test_loss = loss
        self.show_best_calibration(obj, cp_file, self.small_left, self.small_right, self.left, stereo_unrect, mono_out)
        self.calibration_model = self.get_calibration_model()
        self.im_cnt = 0

