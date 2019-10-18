import torch
import pickle
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from train_cont_depth import Dfd, DfdConfig, RES_OUT, Net
from PIL import Image


if __name__ == '__main__':
    torch.cuda.empty_cache()
    np.set_printoptions(linewidth=320)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = DfdConfig(image_size='big', batch_size=1, mode='segmentation', target_mode='cont',lr=0.0001, get_reduced_dataset=False, num_classes=16,dataset='ours', end_epoch=300)
    net = Net(config=config, device=device, num_class=config.num_classes,mode=config.mode, channels=64,skip_layer=True)
    dfd = Dfd(config=config, net=net, device=device, train=False)
    dfd.resume(resume_path=os.path.join(RES_OUT, 'best_model_seg_big_94pm1.pt'))
    num_of_checkpoints = len(os.listdir('resources_out'))
    with open('out_15.raw', 'rb') as f:
        img = pickle.load(f)
    f.close()
    img = Image.fromarray(img)
    img = img.resize((2976,2976))
    img = np.array(img)
    img = dfd.prepare_for_net(img)
    time_before = time.time()
    with torch.no_grad():
        img1_predict = dfd.net(img)
    print ("Time for inference: ", time.time() - time_before)
    if dfd.config.target_mode == 'discrete':
        img1_predict = torch.squeeze(torch.argmax(img1_predict, dim=1),0)

    plt.figure(1)
    plt.imshow(img1_predict, cmap='jet')
    plt.show()
    print ("Done!")