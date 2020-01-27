from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from models.dfd import Dfd_net,psi_to_depth
from models.stackhourglass import PSMNet

from ImageLoader import myImageloader
from local_utils import load_model, scale_invariant
import time
from configurable_stn_projective import ConfigNet
from configurable_stn_projective_both_images import ConfigNet as ConfigNetLeftStn
from stn import Net
import math

from PIL import Image
from local_utils import disparity2depth
from unet.predict_cont import predict_full_img, get_Unet
from MiDaS import MonoDepthNet


def get_dirs(scene_name):
    if scene_name == 'Demo':
        right_train_dir = '/home/yotamg/ForDemo/Right'
        left_train_dir = '/home/yotamg/ForDemo/Left'
    elif scene_name == 'Dynamic':
        right_train_dir = '/home/yotamg/ForDynamic/Right'
        left_train_dir = '/home/yotamg/ForDynamic/Left'
    elif scene_name == 'Outdoor_one':
        # Outdoor one image
        right_train_dir = '/media/yotamg/Yotam/Stereo/Outdoor_one/Right'
        left_train_dir = '/media/yotamg/Yotam/Stereo/Outdoor_one/Left'
    elif scene_name == 'Outdoor_one_rectified':
        # Outdoor one image
        right_train_dir = '/media/yotamg/Yotam/Stereo/Outdoor_one/rectified/Right'
        left_train_dir = '/media/yotamg/Yotam/Stereo/Outdoor_one/rectified/Left'
    elif scene_name == 'Outdoor_one_2':
        # Outdoor one image
        right_train_dir = '/media/yotamg/Yotam/Stereo/Outdoor_one_2/Right'
        left_train_dir = '/media/yotamg/Yotam/Stereo/Outdoor_one_2/Left'
    return right_train_dir, left_train_dir


right_train_dir, left_train_dir = get_dirs('Outdoor_one_rectified')

# right_train_filelist = [os.path.join(right_train_dir, img) for img in os.listdir(right_train_dir) if img.endswith('.png') or img.endswith('.tif') or img.endswith('.bmp')]
# left_train_filelist = [img.replace(right_train_dir, left_train_dir).replace('R', 'L') for img in right_train_filelist]


fig_output_dir = '/home/yotamg/PycharmProjects/affine_transform/figures/calibration_images/'

def get_small_mono(mono_out, device):
    mono_out_normalized = (mono_out - torch.min(mono_out)) / (torch.max(mono_out) / torch.min(mono_out))
    mono_out_small = transforms.ToPILImage()(mono_out_normalized[0].cpu())
    mono_out_small = mono_out_small.resize((256, 256), resample=Image.LANCZOS)
    mono_out_small = transforms.ToTensor()(mono_out_small).to(device)
    mono_out = ((mono_out_small) * (torch.max(mono_out) / torch.min(mono_out)) + torch.min(mono_out))
    return mono_out

def get_mask(stereo_out, right_transformed):
    mono_mask = (stereo_out > 0.5) & (stereo_out < 4.5)
    mask = (right_transformed != 0)[:, 0, :, :]
    mask = mask & mono_mask
    nan_mask = ((torch.isnan(right_transformed) - 1) / 255)[:, 0, :, :]
    mask = mask & nan_mask
    return mask

im_cnt = 0

def show_depth_maps(left, right_transformed, mono, stereo_rect, stereo, blocking=False):
    global im_cnt, fig, ax_list, loss_list
    vmin = torch.min(stereo_rect).item()
    vmax = torch.max(stereo_rect).item()
    # plt.figure(figsize=(18,6))
    if im_cnt == 0:
        ax_list[0].imshow(left[0].permute(1,2,0).detach().cpu())
        plt.setp(ax_list[0].get_xticklabels(), visible=False)
        plt.setp(ax_list[0].get_yticklabels(), visible=False)
        ax_list[0].tick_params(axis='both', which='both', length=0)
        ax_list[0].set_title('Left Image')

        ax_list[1].imshow(right_transformed[0].permute(1,2,0).detach().cpu(), cmap='jet', vmin=vmin, vmax=vmax)
        plt.setp(ax_list[1].get_xticklabels(), visible=False)
        plt.setp(ax_list[1].get_yticklabels(), visible=False)
        ax_list[1].tick_params(axis='both', which='both', length=0)
        ax_list[1].set_title('Right (transformed) Image')

        ax_list[2].imshow(mono[0].detach().cpu(), cmap='jet', vmin=vmin, vmax=vmax)
        plt.setp(ax_list[2].get_xticklabels(), visible=False)
        plt.setp(ax_list[2].get_yticklabels(), visible=False)
        ax_list[2].tick_params(axis='both', which='both', length=0)
        ax_list[2].set_title('Monocular Depth Map')

        ax_list[3].imshow(stereo_rect[0].detach().cpu(),cmap='jet', vmin=vmin, vmax=vmax)
        plt.setp(ax_list[3].get_xticklabels(), visible=False)
        plt.setp(ax_list[3].get_yticklabels(), visible=False)
        ax_list[3].tick_params(axis='both', which='both', length=0)
        ax_list[3].set_title('Stereo Before Calibration')

        ax_list[4].imshow(stereo[0].detach().cpu(),cmap='jet', vmin=vmin, vmax=vmax)
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
    im_cnt += 1
    # plt.show()

def get_mono_and_unrect_stereo(stereo_model, mono_net, midas_model, left, small_left, unet, dfd_net, device, small_right):
    global mono_out
    stereo_model.eval()
    with torch.no_grad():
        if mono_net == 'midas':
            mono_out = midas_model.forward(small_left)
            mono_out = torch.squeeze(mono_out, 0)
        elif mono_net == 'phase-mask':
            dfd_mono_out, _ = dfd_net(left, focal_point=1.5)
            mono_out = torch.unsqueeze(dfd_mono_out, 0)
        else:
            mono_out_unet = predict_full_img(unet, left, device=device)
            mono_out_unet = psi_to_depth(mono_out_unet, focal_point=1.5)
            mono_out = torch.unsqueeze(mono_out_unet, 0)

        _, stereo_unrect = stereo_model(small_left, small_right)
        stereo_unrect = 100 / stereo_unrect
    if mono_net == 'midas':
        mono_out -= torch.min(mono_out)
        mono_out = torch.clamp(1 / mono_out, 0, 3)
    else:
        mono_out = get_small_mono(mono_out, device)
    return stereo_unrect


def train(model, stereo_model, optimizer, small_left, small_right, mono_net, left, stereo_unrect):
    global mono_out,fig, ax_list, loss_list
    model.train()
    stereo_model.eval()
    optimizer.zero_grad()
    stereo_out, theta, right_transformed = model(small_left, small_right)
    stereo_out = 100 / stereo_out

    if mono_net == 'midas':
        mono_out_for_train = mono_out * (torch.mean(stereo_out[stereo_out < 3.0]) / torch.mean(mono_out[mono_out < 3.0]))
    else:
        mono_out_for_train = mono_out

    mask = get_mask(stereo_out, right_transformed)
    loss = F.l1_loss(stereo_out[mask], mono_out_for_train[mask])
    loss_list.append(loss)

    show_depth_maps(left, right_transformed, mono_out_for_train, stereo_unrect, stereo_out)

    loss.backward()
    optimizer.step()

    return loss

def show_best_calibration(cp_file, model, small_left, small_right, mono_net, left, stereo_unrect):
    model.train()
    state_dict = torch.load(cp_file)
    model.load_state_dict(state_dict)
    with torch.no_grad():
        stereo_out, theta, right_transformed = model(small_left, small_right)
    stereo_out = 100 / stereo_out
    if mono_net == 'midas':
        mono_out_for_train = mono_out * (
                    torch.mean(stereo_out[stereo_out < 3.0]) / torch.mean(mono_out[mono_out < 3.0]))
    else:
        mono_out_for_train = mono_out
    show_depth_maps(left, right_transformed, mono_out_for_train, stereo_unrect, stereo_out, blocking=True)


def main(l_img=None, r_img=None):
    global loss_list,ax_list,fig
    torch.cuda.empty_cache()
    device = torch.device('cuda:0')

    midas_model = None
    dfd_net = None
    unet = None
    mono_net = 'phase-mask'
    # mono_net = 'midas'
    # mono_net = 'unet  '

    stereo_model = PSMNet(192, device=device, dfd_net=False, dfd_at_end=False, right_head=False)
    stereo_model = nn.DataParallel(stereo_model)
    stereo_model.cuda()
    state_dict = torch.load('checkpoints/PSM/pretrained_model_KITTI2015.tar')
    # state_dict = torch.load('/home/yotamg/PycharmProjects/PSMNet_orig/checkpoint_300.tar')
    stereo_model.load_state_dict(state_dict['state_dict'], strict=False)
    stereo_model.train()

    stereo_model2 = PSMNet(192, device=device, dfd_net=False, dfd_at_end=False, right_head=False)
    stereo_model2 = nn.DataParallel(stereo_model2)
    stereo_model2.cuda()
    state_dict = torch.load('checkpoints/PSM/pretrained_model_KITTI2015.tar')
    # state_dict = torch.load('/home/yotamg/PycharmProjects/PSMNet_orig/checkpoint_300.tar')
    stereo_model2.load_state_dict(state_dict['state_dict'], strict=False)
    stereo_model2.train()

    if mono_net == 'phase-mask':
        dfd_net = Dfd_net(mode='segmentation', target_mode='cont', pool=False)
        dfd_net = dfd_net.eval()
        dfd_net = dfd_net.to(device)
        load_model(dfd_net, device, model_path='checkpoints/Dfd/checkpoint_257.pth.tar')

    elif mono_net == 'midas':
        # load network
        midas_model_path = 'checkpoints/Midas/model.pt'
        midas_model = MonoDepthNet(midas_model_path)
        midas_model.to(device)
        midas_model.eval()
    else:
        unet = get_Unet('models/unet/CP100_w_noise.pth', device=device)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    if l_img is not None:
        right_train_filelist = [r_img]
        left_train_filelist = [l_img]
    else:
        right_train_filelist = ['Sample_Images/R_10.tif']
        left_train_filelist = ['Sample_Images/L_10.tif']

    patch_size = 256

    train_db = myImageloader(left_img_files=left_train_filelist, right_img_files=right_train_filelist, supervised=False,
                             train_patch_w=patch_size,
                             transform=transforms.Compose(
                                 [transforms.ToTensor()]),
                             label_transform=transforms.Compose([transforms.ToTensor()]), get_filelist=True)

    train_loader = torch.utils.data.DataLoader(train_db, batch_size=1, shuffle=True, num_workers=0)

    # model = Net(stereo_model=stereo_model).to(device)
    model = ConfigNet(stereo_model=stereo_model2, stn_mode='projective', ext_disp2depth=False, device=device).to(device)

    if mono_net == 'midas':
        # lr = 0.0001
        lr = 0.001
    else:
        lr = 0.001

    optimizer = optim.SGD(model.parameters(), lr=lr)
    # optimizer = optim.Adam(model.parameters(), lr=lr)

    for param in model.stereo_model.parameters():
        param.requires_grad = False

    show_train_images = False
    test_show_images = False

    num_of_epochs = 50
    # model = ConfigNet(stereo_model=stereo_model2, stn_mode='projective', ext_disp2depth=False).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for param in model.stereo_model.parameters():
        param.requires_grad = False
    for batch_idx, (left, right, small_left, small_right) in enumerate(train_loader):
        left, right, small_left, small_right = left.to(device), right.to(device), small_left.to(device), small_right.to(device)

    plt.ion()
    fig, ax_list = plt.subplots(3, 2, figsize=(18,6))
    fig.tight_layout()
    ax_list = ax_list.ravel()
    loss_list = list()

    # midas_model = None

    stereo_unrect = get_mono_and_unrect_stereo(stereo_model, mono_net, midas_model, left, small_left, unet, dfd_net, device, small_right)


    dir_checkpoint = 'checkpoints/demo_cp'
    best_test_loss = 2.0
    epoch = 1
    should_finish = False
    start_over = False
    while not should_finish:
    # for epoch in range(1, num_of_epochs + 1):
        loss = train(model, stereo_model, optimizer, small_left, small_right, mono_net, left, stereo_unrect)
        if epoch == 1:
            first_loss = loss
        if start_over:
            if epoch % 10 == 9:
                if best_test_loss >= 0.5 * first_loss:
                    #Start over
                    model = ConfigNet(stereo_model=stereo_model2, stn_mode='projective', ext_disp2depth=False).to(device)
                    optimizer = optim.SGD(model.parameters(), lr=lr)
                    for param in model.stereo_model.parameters():
                        param.requires_grad = False
                    best_test_loss = 2.0
                    epoch = 1
                    print ("Start Over")
        if loss < best_test_loss:
            cp_file = os.path.join(dir_checkpoint, 'CP{}.pth'.format(epoch))
            torch.save(model.state_dict(),
                       cp_file)
            print('Checkpoint {} saved !'.format(epoch))
            best_test_loss = loss
        epoch += 1
        should_finish = epoch == num_of_epochs
    show_best_calibration(cp_file, model, small_left, small_right, mono_net, left, stereo_unrect)

if __name__ == '__main__':
    main()