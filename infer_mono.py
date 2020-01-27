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

midas_model_path = 'checkpoints/Midas/model.pt'
mono_model = MonoDepthNet(midas_model_path)
mono_model.to(device)
mono_model.eval()

img_dir = '/media/yotamg/Assaf/phone_images/Jan20/2020_01_22'

out_dir = '/media/yotamg/Yotam/depth'

im_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('jpg') or f.endswith('png') or f.endswith('tif')]

for img_file in im_list:
    try:
        orig_img = Image.open(img_file)
    except:
        continue
    w, h = orig_img.size
    orig_img = orig_img.resize((w//8,h//8))
    # orig_img = orig_img[:448, :608,:]
    img = transforms.ToTensor()(orig_img).to(device)
    # img = img.permute((2,0,1))
    img = torch.unsqueeze(img, 0)
    try:
        out = mono_model(img).detach().cpu().numpy()[0][0]
    except:
        print ("A")
    with open(img_file.replace(img_dir, out_dir).replace('.tif', '_depth_256x256.pickle'), 'wb') as f:
        pickle.dump(out, f)
    # plt.imsave(os.path.join(img_dir, img_file.replace('.jpg', '_depth.jpg')), out)
    plt.subplot(121)
    plt.imshow(orig_img)
    plt.subplot(122)
    plt.imshow(out, cmap='jet')
    plt.show()