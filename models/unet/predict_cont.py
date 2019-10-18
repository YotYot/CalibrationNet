import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from unet import UNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks
from data_vis import plot_img_and_mask

from torchvision import transforms
from sintel_io import depth_read

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', '-m', default='checkpoints/cont_cp/CP48.pth',
    parser.add_argument('--model', '-m', default='CP100_w_noise.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def predict_img(net,
                full_img,
                device):
    net.eval()
    img = resize_and_crop(full_img, scale=1)
    img = normalize(img)
    img = hwc_to_chw(img)
    X_img = torch.from_numpy(img).unsqueeze(0)
    X_img = X_img.to(device)

    with torch.no_grad():
        output_img = net(X_img)
    return output_img

def predict_full_img(net, img, device):
    x_part_margin = 0
    part_margin = 128
    margin = 256
    patch_w = 2048 - 256
    patch_h = 2048 - 256
    img = transforms.ToPILImage()(img[0].cpu())
    mask = torch.zeros((img.size[1], img.size[0])).to(device)
    x0 = 0
    for i in range((img.size[0] // (patch_w - margin)) + 1):
        y0 = 0
        y_part_margin = 0
        for j in range((img.size[1] // (patch_h - margin)) + 1):
            x1 =  min(img.size[0], x0+patch_w)
            y1 =  min(img.size[1], y0+patch_h)
            img_part = img.crop((x0,y0,x1,y1))
            mask_part = predict_img(net=net,
                                    full_img=img_part,
                                    device=device)

            mask = implant_mask_part(img, mask, mask_part, x0, x1, x_part_margin, y0, y1, y_part_margin)
            y_part_margin = part_margin
            y0 += patch_h - margin
        x_part_margin = part_margin
        x0 += patch_w - margin
    return mask


def implant_mask_part(img, mask, mask_part, x0, x1, x_part_margin, y0, y1, y_part_margin):
    mask_part = torch.squeeze(mask_part,0)
    if y1 == img.size[1] and x1 == img.size[0]:
        mask[y0 + y_part_margin:y1, x0 + x_part_margin:x1] = \
            mask_part[y_part_margin:mask_part.shape[0], x_part_margin:mask_part.shape[1]]
    elif y1 == img.size[1]:
        mask[y0 + y_part_margin:y1, x0 + x_part_margin:x1 - x_part_margin] = \
            mask_part[y_part_margin:mask_part.shape[0],
            x_part_margin:mask_part.shape[1] - x_part_margin]
    elif x1 == img.size[0]:
        mask[y0 + y_part_margin:y1 - y_part_margin, x0 + x_part_margin:x1] = \
            mask_part[y_part_margin:mask_part.shape[0] - y_part_margin,
            x_part_margin:mask_part.shape[1]]
    else:
        mask[y0 + y_part_margin:y1 - y_part_margin, x0 + x_part_margin:x1 - x_part_margin] = \
            mask_part[y_part_margin:mask_part.shape[0] - y_part_margin,
            x_part_margin:mask_part.shape[1] - x_part_margin]
    return mask


def get_Unet(model_cp, device):
    net = UNet(n_channels=3, n_classes=16, cont=True)
    print("Loading model {}".format(model_cp))
    net.to(device)
    net.load_state_dict(torch.load(model_cp, map_location=device))
    print("Model loaded !")
    return net


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    in_dir = '/media/yotamg/Yotam/Real-Images/png'
    # in_dir = '/media/yotamg/Yotam/Stereo/Indoor/Left/'
    # in_dir = '/media/yotamg/Yotam/Stereo/Outdoor/Left/'
    in_files = [os.path.join(in_dir, img) for img in os.listdir(in_dir)]
    # in_files = ['/media/yotamg/Yotam/Real-Images/png/Scene3_out_9.png']
    out_files = get_output_filenames(args)

    net = get_Unet()

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)
        if img.size[0] < img.size[1]:
            print("Error: image height larger than the width")
        mask = predict_full_img(net, img)

        if args.viz:
            print("Visualizing results for image {}, close to continue ...".format(fn))
            # mask[mask==15] = 1
            gt = None
            gt_file = fn.replace('rgb', 'cont_GT').replace('_1500_maskImg.png', '_GT.dpt')
            if gt_file.endswith('.dpt'):
                gt = depth_read(gt_file)
                gt += 5
            mask += 5
            plot_img_and_mask(img, mask, gt, cont=True)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])

            print("Mask saved to {}".format(out_files[i]))
