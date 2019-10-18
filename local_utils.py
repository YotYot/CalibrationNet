import os
import shutil
from os import path
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import sintel_io
from sintel_io import depth_read
# import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

def projective_transform(img, theta):
    img = transforms.ToTensor()(img)
    img_shape = img.size()
    tx = torch.linspace(-1, 1, img_shape[2]).unsqueeze(0).repeat(img_shape[1], 1)
    ty = torch.linspace(-1, 1, img_shape[1]).unsqueeze(1).repeat(1, img_shape[2])
    grid = torch.zeros((1, img_shape[1], img_shape[2], 3), dtype=torch.float32)
    grid[0, :, :, 0] = tx
    grid[0, :, :, 1] = ty
    grid[0, :, :, 2] = torch.ones(img_shape[1], img_shape[2])
    grid = torch.mm(grid.reshape(-1, 3), theta.t()).reshape(1, img_shape[1], img_shape[2], 3)
    grid[0, :, :, 0] = grid[0, :, :, 0].clone() / grid[0, :, :, 2].clone()
    grid[0, :, :, 1] = grid[0, :, :, 1].clone() / grid[0, :, :, 2].clone()
    img = torch.unsqueeze(img, 0)
    img = F.grid_sample(img, grid[:, :, :, :2])
    img = torch.squeeze(img, 0)
    img = transforms.ToPILImage()(img)
    return img


def depth2disparity(depth, device):
    imSz = torch.Tensor([depth.shape])[0]
    f = 24.0  # focal length in mm
    sensor_w = 32.0  # Sensor width in mm
    num_of_pixels = 512  # number of pixels in horizontal direction
    pixel_sz = sensor_w / num_of_pixels
    B = 100  # distance between the two sensors in mm
    cnt = (torch.floor(imSz / 2) + 1)
    xi = torch.arange(imSz[2]) - cnt[2]
    yi = torch.arange(imSz[3]) - cnt[3]
    Xi, Yi = torch.meshgrid([xi, yi])
    Ri = pixel_sz * torch.sqrt((Xi ** 2) + (Yi ** 2) + (f / pixel_sz) ** 2)
    Ri = torch.unsqueeze(Ri, 0).repeat(depth.shape[0], 1, 1)
    f_Ri = (f / Ri).to(device)
    calc_depth = depth * 1e3 * f_Ri
    disp = ((B * f) / calc_depth) // pixel_sz
    return disp

def disparity2depth(disp, device=None):
    device = device if device else disp.get_device()
    imSz = torch.Tensor([disp.shape])[0]
    f = 24.0  # focal length in mm
    sensor_w = 32.0  # Sensor width in mm
    num_of_pixels = 512  # number of pixels in horizontal direction
    pixel_sz = sensor_w / num_of_pixels
    B = 100  # distance between the two sensors in mm
    cnt = (torch.floor(imSz / 2) + 1)
    xi = torch.arange(imSz[2]) - cnt[2]
    yi = torch.arange(imSz[3]) - cnt[3]
    Xi, Yi = torch.meshgrid([xi, yi])
    Ri = pixel_sz * torch.sqrt((Xi ** 2) + (Yi ** 2) + (f / pixel_sz) ** 2)
    Ri = torch.unsqueeze(Ri, 0).repeat(disp.shape[0], 1, 1)
    f_Ri = (f / Ri).to(device)
    calc_depth = ((B * f) / (disp * pixel_sz))
    depth = calc_depth / (1e3 * f_Ri)
    return depth

def load_model(model, device, model_path):
    print("loading checkpoint from: ", model_path)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

def noisy(image, sigma=0.0235):
    row, col, ch = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    return gauss

def flatten_dir(src_dir, dst_dir):
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    for dir in os.listdir(src_dir):
        for file in os.listdir(os.path.join(src_dir,dir)):
            new_name = dir + '_' + file
            orig_path = os.path.join(src_dir, dir, file)
            new_path = os.path.join(dst_dir, new_name)
            shutil.copy(orig_path, new_path)

def dir_tif2png(src_dir, dst_dir):
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    for lbl in os.listdir(src_dir):
        if lbl.endswith(".tif"):
            filepath = path.join(src_dir, lbl)
            img = Image.open(filepath)
            img_arr = np.array(img)
            img_int = (img_arr / 256).astype(np.uint8)
            # img_rgb = cv2.cvtColor(img_int, cv2.COLOR_BAYER_BG2RGB)
            base = os.path.splitext(lbl)[0]
            out_file = base + ".png"
            Image.fromarray(img_rgb).save(path.join(dst_dir, out_file), compress_level=0)

def mv_percent_for_testing(train_dir=None, test_dir=None, percentage=30):
    file_list = os.listdir(train_dir)
    nof_files = len(file_list)
    files_for_test = round((float(percentage) / 100) * nof_files)
    files_for_test = np.random.choice(file_list, int(files_for_test),replace=False)
    if not path.isdir(test_dir):
        os.makedirs(test_dir)
    for file in files_for_test:
        src = path.join(train_dir, file)
        dst = path.join(test_dir, file)
        os.rename(src, dst)

def raw2png(src_dir, dst_dir):
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    for raw_img in os.listdir(src_dir):
        raw_path = os.path.join(src_dir,raw_img)
        with open(raw_path,'rb') as f:
            img = pickle.load(f)
            img_name = raw_img.split(".")[0]+".png"
            img_path = os.join.path(dst_dir, img_name)
            with open(img_path,'wb') as f:
                Image.fromarray(img).save(f, compress_level=0)

def get_depth_histogram(depth_dir):
    all_d = list()
    for file in os.listdir(depth_dir):
        d = depth_read(os.path.join(depth_dir,file))
        all_d.append(np.concatenate(d))
    np.histogram(all_d)

def get_histo_for_discrete_depth(train_dir):
    histo = np.zeros(15, dtype=np.int32)
    for patch in os.listdir(train_dir):
        if patch.endswith('.dpt'):
            patch_path = path.join(train_dir, patch)
            dpt = depth_read(patch_path)
            dpt_histo = np.bincount(np.concatenate(dpt).astype(int), minlength=16)[1:]
            histo += dpt_histo
    return histo

def remove_mask_from_filename(dir):
    for file in os.listdir(dir):
        new_file = file.replace('_maskImg','')
        shutil.move(os.path.join(dir,file), os.path.join(dir,new_file))


def get_depth_histogram(depth_dir):
    dpt_list = list()
    for depth in os.listdir(depth_dir):
        path = os.path.join(depth_dir,depth)
        dpt = depth_read(path)
        dpt_list.append(dpt)
    dpt_list = np.concatenate(np.concatenate(dpt_list))
    histo = np.histogram(dpt_list, np.arange(1,np.max(dpt_list)))
    plt.plot(histo[0])

# import shutil

def move_same(src_dir, dst_dir):
    for img in os.listdir(src_dir):
        # img = img.replace('.tif', '_1100_maskImg.png')
        if img.endswith('.png') or img.endswith('.tif'):
            img = img.replace('.tif', '_1500_maskImg.png')
            img_path = os.path.join(dst_dir, img)
            new_img_dir = os.path.join(dst_dir, 'val')
            new_img_path = os.path.join(new_img_dir,img)
            if not os.path.isdir(new_img_dir):
                os.makedirs(new_img_dir)
            shutil.move(img_path, new_img_path)

def replace_pattern(dir, src_pattern, target_pattern):
    for file in os.listdir(dir):
        target_file = file.replace(src_pattern, target_pattern)
        target_path = os.path.join(dir, target_file)
        src_path = os.path.join(dir, file)
        shutil.move(src_path, target_path)



def scale_invariant(output, target):
    # di = output - target
    di = target - output
    n = (256 * 256)
    di2 = torch.pow(di, 2)
    fisrt_term = torch.sum(di2,(1,2))/n
    second_term = torch.pow(torch.sum(di,(1,2)), 2)/ (n**2)
    loss = fisrt_term - second_term
    return loss.mean()


def example():
    # dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Tau-agent/oof_flatten'
    # for file in os.listdir(dir):
    #     filepath = os.path.join(dir, file)
    #     os.rename(filepath, filepath.replace('_L',''))

    # flatten_dir(
    #     '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Tau-Stereo_orig_structure/Headbutt_L/Occlusions',
    #     '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Tau-agent/occ_flatten')
    # mv_percent_for_testing('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Stereo/Tau_right_images/right_images_clean/', '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Stereo/Tau_right_images/right_images_clean/val', 20)
    # move_same('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Tau-agent/right_images/right_images_filtered/val', '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Stereo/Tau_right_images/right_images_clean')
    # move_same('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Tau-agent/right_images/right_images_clean/val', '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Stereo/Tau_right_images/dn1500_D5/rgb')
    replace_pattern('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Stereo/Tau_left_images/original_disp/', '_L_', '_')

if __name__ == '__main__':
   example()

# get_depth_histogram('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Stereo/Tau_left_images/original_depth/')
# remove_mask_from_filename('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/clean_left_filtered_alley_1')
# remove_mask_from_filename('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/clean_left_filtered_all_but_alley_1')
# remove_mask_from_filename('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/clean_left_flatten_filter_testing')
# remove_mask_from_filename('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/clean_left_flatten_all_no_lens/rgb/')
# remove_mask_from_filename('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/left_filtered_adapted/rgb')
# remove_mask_from_filename('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Tau-agent/Filtered_images')
# a =  (get_histo_for_discrete_depth('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/clean_left_flatten_all_filter/GT'))
# plt.plot(a)
# print (a)
# get_depth_histogram('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/depth_flatten')
# flatten_dir('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/clean_left_alley_1/', '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/clean_left_alley_1/')
# flatten_dir('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/clean_left_all_but_alley_1/', '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/clean_left_all_but_alley_1/')
# flatten_dir('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/clean_right', '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/clean_right_flatten')
# flatten_dir('/home/yotamg/data/sintel_depth/training/clean_right', '/home/yotamg/data/sintel_depth/training/clean_right_flatten')
#flatten_dir('/home/yotamg/data/sintel_depth/training/disparities', '/home/yotamg/data/sintel_depth/training/disparities_flatten')
# flatten_dir('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Real-Images', '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Real-Images')
# dir_tif2png('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Real-Images/', '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Real-Images/png')

# mv_percent_for_testing('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/clean_left_flatten', '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/clean_left_flatten_testing',10)
# mv_percent_for_testing('/home/yotamg/data/sintel_depth/training/clean_right_flatten', '/home/yotamg/data/sintel_depth/testing/clean_right_flatten',20)
# mv_percent_for_testing('/home/yotamg/data/sintel_depth/training/disparities_flatten', '/home/yotamg/data/sintel_depth/testing/disparities_flatten',20)
# mv_percent_for_testing('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/clean_left_flatten_filter_training', '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/clean_left_flatten_filter_testing',20)
# raw2png('/home/yotamg/data/raw_rgb_images', '/home/yotamg/data/raw_rgb_pngs')
# import shutil
# dir = '/home/yotamg/data/raw_rgb_pngs/sintel_only'
# for file in os.listdir(dir):
#     if file.endswith(".png"):
#         p = os.path.join(dir,file)
#         file_splits = file.split("_")
#         name = file_splits[0] + "_" + file_splits[1] + "_" + file_splits[2] + "_" + file_splits[3] + ".png"
#         shutil.move(p, os.path.join(dir,name))