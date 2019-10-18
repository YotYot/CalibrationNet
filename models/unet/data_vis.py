import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def plot_img_and_mask(img, mask, gt=None, cont=False):
    if gt is not None:
        fig_num = 3
    else:
        fig_num = 2
    fig = plt.figure(figsize=(12,6))
    a = fig.add_subplot(1, fig_num, 1)
    a.set_title('Input image')
    plt.imshow(img)


    b = fig.add_subplot(1, fig_num, 2)
    b.set_title('Output mask')
    plt.imshow(mask, vmin=1, vmax=15, cmap='jet')

    if gt is not None:
        b = fig.add_subplot(1, fig_num, 3)
        if cont:
            loss = np.mean(np.abs(mask - gt))
            b.set_title('GT mask, loss: {}'.format(loss))
        else:
            acc = np.sum(mask == gt) / (mask.shape[0] * mask.shape[1])
            b.set_title('GT mask, acc: {}'.format(acc))
        plt.imshow(gt, vmin=1, vmax=15)
    plt.show()