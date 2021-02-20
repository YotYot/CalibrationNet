import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import pickle

import matplotlib.pyplot as plt
from PIL import Image
import os

from models.dfd import Dfd_net,psi_to_depth
from models.stackhourglass import PSMNet
from models.MiDaS import MonoDepthNet
from unet.predict_cont import predict_full_img, get_Unet
from configurable_stn_projective import ConfigNet
from ImageLoader import myImageloader

device = torch.device('cuda:1')

# midas_model_path = 'checkpoints/Midas/model.pt'
# mono_model = MonoDepthNet(midas_model_path)
# mono_model.to(device)
# mono_model.eval()

def get_stereo_model():
        stereo_model = PSMNet(192, device=device, dfd_net=False, dfd_at_end=False, right_head=False)
        stereo_model = nn.DataParallel(stereo_model)
        stereo_model.cuda()
        state_dict = torch.load('checkpoints/PSM/pretrained_model_KITTI2015.tar')
        stereo_model.load_state_dict(state_dict['state_dict'], strict=False)
        stereo_model.eval()
        return stereo_model


stereo_model = get_stereo_model()


# img_dir = '/media/yotamg/Assaf/phone_images/Jan20/2020_01_22'
# img_dir = '/media/yotamg/Yotam/data/clean_jpg_images/'
img_dir = '/media/yotamg/Yotam/Stereo/Tau_left_images/dn700/'

out_dir = '/home/yotam/stereo_depth/'

im_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('jpg') or f.endswith('png') or f.endswith('tif')]

for img_file in im_list:
    try:
        left_img = Image.open(img_file)
        right_img_file = img_file.replace('left', 'right').replace('_700_maskImg.png', '.tif').replace('dn700', 'right_images_clean')
        right_img_file = right_img_file.split('/')
        just_file = right_img_file[-1].split('_')[0] + '_R_' + right_img_file[-1].split('_')[1]
        right_img_file = '/'.join(right_img_file[:-1]) + '/' + just_file
        right_img = Image.open(right_img_file)
    except:
        continue
    w, h = left_img.size
    # orig_img = orig_img.resize((w//8,h//8))
    # orig_img = orig_img[:448, :608,:]
    left_img = left_img.resize((w // 2, h // 2))
    right_img = right_img.resize((w // 2, h // 2))
    left_img = transforms.ToTensor()(left_img).to(device)
    right_img = transforms.ToTensor()(right_img).to(device)
    # img = img.permute((2,0,1))
    left_img = torch.unsqueeze(left_img, 0)
    right_img = torch.unsqueeze(right_img, 0)
    # try:
    _, out = stereo_model(left_img, right_img)
    out = 100 / out
        # out = mono_model(img)[0][0].detach().cpu().numpy()
    # except:
    #     print ("A")
    with open(img_file.replace(img_dir, out_dir).replace('.png', '_depth.pickle'), 'wb') as f:
        pickle.dump(out, f)
    # plt.imsave(os.path.join(img_dir, img_file.replace('.jpg', '_depth.jpg')), out)
    # plt.subplot(121)
    # plt.imshow(left_img[0].detach().cpu().numpy().transpose(1,2,0))
    # plt.subplot(122)
    # plt.imshow(out[0].detach().cpu().numpy(), cmap='jet')
    # plt.show()