#!/usr/bin/env python

#------------------------------------------------------------------------------
#                 PyuEye example - main modul
#
# Copyright (c) 2017 by IDS Imaging Development Systems GmbH.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#------------------------------------------------------------------------------

from pyueye_example_camera import Camera
from pyueye_example_utils import FrameThread
from pyueye_example_gui import PyuEyeQtApp, PyuEyeQtView
from PyQt4 import QtGui

import torch
import pickle
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from train_cont_depth import Dfd, DfdConfig, RES_OUT, Net
from PIL import Image
import cv2

from pyueye import ueye

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = DfdConfig(image_size='big', batch_size=1, mode='segmentation', target_mode='cont',num_classes=16)
net = Net(config=config, device=device, num_class=config.num_classes, mode=config.mode, channels=64,
      skip_layer=True)
dfd = Dfd(config=config, net=net, device=device, train=False)

# Switch bwetween those lines for the different models
# dfd.resume(resume_path=os.path.join(RES_OUT, 'best_model_seg_big_image_89pm1.pt')) # Original model
dfd.resume() # Trained model without input skip connection

cmap = plt.cm.ScalarMappable(cmap='jet')

def get_prediction(self, image_data):
    img = image_data.as_1d_image()
    img = dfd.prepare_for_net(img)
    time_before = time.time()
    with torch.no_grad():
        img1_predict = dfd.net(img)
    print("Time for inference: ", time.time() - time_before)
    if dfd.config.target_mode == 'discrete':
        img1_predict = torch.squeeze(torch.argmax(img1_predict, dim=1), 0)
    time_before = time.time()
    img1_predict = cmap.to_rgba(img1_predict.detach().cpu(), bytes=True)[:,:,:3]
    print("Time for Proccessing3: ", time.time() - time_before)
    img1_predict = np.array(img1_predict, dtype=np.uint8)
    print("Time for Proccessing4: ", time.time() - time_before)
    return QtGui.QImage(img1_predict.data,img1_predict.shape[1],img1_predict.shape[0],QtGui.QImage.Format_RGB888)

def process_image(self, image_data):
    # reshape the image data as 1dimensional array
    image = image_data.as_1d_image()
    # show the image with Qt
    # plt.imsave('test.png', np.array(image))
    return QtGui.QImage(image.data,image_data.mem_info.width,image_data.mem_info.height,QtGui.QImage.Format_RGB888)
    # return QtGui.QImage(big_image.data,big_image.shape[1],big_image.shape[0],QtGui.QImage.Format_RGB888)

def main():

    # we need a QApplication, that runs our QT Gui Framework
    app = PyuEyeQtApp()

    # a basic qt window
    view = PyuEyeQtView()
    view_depth = PyuEyeQtView()
    view_depth.show()
    view.show()
    view.user_callback = process_image
    view_depth.user_callback = get_prediction

    # camera class to simplify uEye API access
    cam = Camera()
    cam.init()
    cam.set_colormode(ueye.IS_CM_RGB8_PACKED)
    #cam.set_colormode(ueye.IS_CM_SENSOR_RAW8)
    cam.set_aoi(0,0, 4896, 3680)
    cam.set_autoexposure_enabled(False)
    cam.set_autogain_enabled(False)
    #cam.get_rgbGain()
    cam.set_pixelClock(100)
    cam.set_frameRate(2)
    cam.set_exposure(250)
    cam.set_gain(0)

    cam.set_rgbGain([23, 0, 31])
    #cam.setRopEffect()
    cam.alloc()
    cam.capture_video()

    # a thread that waits for new images and processes all connected views
    thread = FrameThread(cam, view)
    thread.start()

    thread2 = FrameThread(cam, view_depth)
    thread2.start()


    # cleanup
    app.exit_connect(thread.stop)
    app.exec_()

    thread.stop()
    thread.join()

    cam.stop_video()
    cam.exit()

if __name__ == "__main__":
    main()

