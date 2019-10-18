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

import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from pyueye_example_camera import Camera
from pyueye_example_utils import FrameThread
from pyueye_example_gui import PyuEyeQtApp, PyuEyeQtView
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import *
import PyQt4.QtCore
from time import sleep
from threading import Thread
import threading

import torch
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# import demo
from pyueye import ueye
from calib_train import CalibTrain

import time
import numpy as np

save_snapshot_0 = False
save_snapshot_1 = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cmap = plt.cm.ScalarMappable(cmap='jet')


def init_cam(device_id, views):
    # camera class to simplify uEye API access
    cam = Camera(device_id=device_id)
    cam.init()
    cam.set_colormode(ueye.IS_CM_RGB8_PACKED)
    # cam.set_colormode(ueye.IS_CM_SENSOR_RAW8)
    cam.set_aoi(0, 0, 4896, 3680)
    cam.set_autoexposure_enabled(False)
    cam.set_autogain_enabled(False)
    # cam.get_rgbGain()
    cam.set_pixelClock(100)
    cam.set_frameRate(0.2)
    cam.set_exposure(250)
    cam.set_gain(0)
    cam.set_rgbGain([23, 0, 31])
    # cam.setRopEffect()
    cam.alloc()
    cam.capture_video()
    # a thread that waits for new images and processes all connected views
    thread = FrameThread(cam, views)
    thread.start()
    return thread, cam


def process_image(obj, image_data, cam_id):
    # print ("{} :Refresh for cam #{}".format(time.time(), cam_id))
    # reshape the image data as 1dimensional array
    image = image_data.as_1d_image()
    if cam_id == 0 or cam_id == 1:
        obj.image_arr = image

    # show the image with Qt

    return QtGui.QImage(image.data,image_data.mem_info.width,image_data.mem_info.height,QtGui.QImage.Format_RGB888)
    # return QtGui.QImage(big_image.data,big_image.shape[1],big_image.shape[0],QtGui.QImage.Format_RGB888)


def get_stereo_prediction(obj, image_data, cam_id):

    # reshape the image data as 1dimensional array
    return QtGui.QImage(image_data.data, image_data.shape[1], image_data.shape[0], QtGui.QImage.Format_RGB888)
    # return QtGui.QImage(big_image.data,big_image.shape[1],big_image.shape[0],QtGui.QImage.Format_RGB888)

def get_calib_prediction(obj, image_data, cam_id):

    # reshape the image data as 1dimensional array
    image_data = cmap.to_rgba(image_data[0], bytes=True)[:, :, :3]
    image_data = np.array(image_data, dtype=np.uint8)
    return QtGui.QImage(image_data.data, image_data.shape[1], image_data.shape[0], QtGui.QImage.Format_RGB888)
    # return QtGui.QImage(big_image.data,big_image.shape[1],big_image.shape[0],QtGui.QImage.Format_RGB888)

def get_np_image(obj, image_data, cam_id):

    # reshape the image data as 1dimensional array
    image_data = np.transpose(image_data[0], (1,2,0))
    image_data = np.clip(image_data,0,1)
    image_data = cmap.to_rgba(image_data, bytes=True)[:, :, :3]
    image_data = np.array(image_data, dtype=np.uint8)
    return QtGui.QImage(image_data.data, image_data.shape[1], image_data.shape[0], QtGui.QImage.Format_RGB888)
    # return QtGui.QImage(big_image.data,big_image.shape[1],big_image.shape[0],QtGui.QImage.Format_RGB888)


class UncalibratedStereo(QWidget, Thread):
    def __init__(self, left_im, right_im):
        QWidget.__init__(self)
        Thread.__init__(self)
        self.running = True
        self.left = left_im
        self.right = right_im
        self.label = QLabel()
        pixmap = QPixmap('/home/yotamg/Downloads/5deg_calibration.png')
        self.label.setPixmap(pixmap)
        layout = QHBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.show()

    def run(self):
        while self.running:
            print(type(self.right.image))
            if self.right.image:
                out = self.right.image
                self.label.setPixmap(QtGui.QPixmap.fromImage(out))
                # self.show()
            sleep(1)

    def stop(self):
        self.running = False


class MonoStereoPredictionThread(Thread):
    def __init__(self, win):
        super(MonoStereoPredictionThread, self).__init__()
        self.win = win
        self.do_predict = False
        self.pause_predict_while_calibrating = False

    def run(self):
        while True:
            if self.do_predict and not self.pause_predict_while_calibrating:
                self.get_prediction()
                self.do_predict = False
            sleep(0.1)

    def get_prediction(self):
        if win.left.image_arr is not None and win.right.image_arr is not None:
            mono_out, stereo_unrect = win.calibTrain.get_mono_prediction(win.left.image_arr, win.right.image_arr)
            mono_out = torch.squeeze(mono_out, 0)
            mono_out = np.array(mono_out.cpu())
            stereo_unrect = torch.squeeze(stereo_unrect, 0)
            stereo_unrect = np.array(stereo_unrect.cpu())
            mono_out = cmap.to_rgba(mono_out, bytes=True)[:, :, :3]
            mono_out = np.array(mono_out, dtype=np.uint8)
            stereo_unrect = cmap.to_rgba(stereo_unrect, bytes=True)[:, :, :3]
            stereo_unrect = np.array(stereo_unrect, dtype=np.uint8)
            win.mono.image = win.stereo.call_back_fn(win, mono_out, 3)
            win.mono.update_signal.emit(win.mono.image)
            win.stereo.image = win.stereo.call_back_fn(win, stereo_unrect, 3)
            win.stereo.update_signal.emit(win.stereo.image)
            torch.cuda.empty_cache()

    def enable_prediction(self):
        self.do_prediction = True

class CalibrateThread(Thread):
    def __init__(self, win, calibTrain):
        super(CalibrateThread, self).__init__()
        self.win = win
        # self.l_img = l_img
        # self.r_img = r_img
        self.calibTrain = calibTrain
        self._stop_event = threading.Event()
        self.do_calibrate = False

    def run(self):
        while True:
            if self.do_calibrate:
                self.cal_flow()
                self.do_calibrate = False
            sleep(0.01)


    def cal_flow(self):
        self.win.left.update = False
        self.win.right.update = False
        c1.freeze_video()
        c2.freeze_video()
        self.calibrate()
        c1.capture_video()
        c2.capture_video()
        calibTrain = CalibTrain(mono_net=mono_net)
        self.calibTrain = calibTrain
        self.win.calibTrain = calibTrain
        # t1, c1 = init_cam(0, win.left)
        # t2, c2 = init_cam(1, win.right)

        self.win.left.update = True
        self.win.right.update = True
        self.win.predictionThread.pause_predict_while_calibrating = False

    def calibrate(self):
        self.calibTrain.best_test_loss = 2.0
        self.win.figure.clf()
        self.calibTrain.calibrate(obj=win, l_img=win.left.image_arr, r_img=win.right.image_arr)



class CalibDisp(QWidget):
    def __init__(self, calibTrain, MainWindow):
        super(CalibDisp, self).__init__()
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1087, 624)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.frame = QtGui.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10, 10, 531, 411))
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName(("frame"))

        self.left = PyuEyeQtView(cam_id=0, callback_fn=process_image, parent=self.frame)
        self.left.setGeometry(QtCore.QRect(10, 10, 252, 192))

        self.right = PyuEyeQtView(cam_id=1, callback_fn=process_image, parent=self.frame)
        self.right.setGeometry(QtCore.QRect(270, 10, 252, 192))

        self.mono = PyuEyeQtView(2, parent=self.frame)
        self.mono.setGeometry(QtCore.QRect(10, 210, 252, 192))

        self.stereo = PyuEyeQtView(3, callback_fn=get_stereo_prediction, parent=self.frame)
        self.stereo.setGeometry(QtCore.QRect(270, 210, 252, 192))

        self.calibTrain = calibTrain
        self.calibrateThread = CalibrateThread(win=self, calibTrain=self.calibTrain)
        self.calibrateThread.start()

        self.predictionThread = MonoStereoPredictionThread(win=self)
        self.predictionThread.start()

        self.button = QtGui.QPushButton(self.centralwidget)
        self.button.clicked.connect(self.start_calib)
        self.button.setGeometry(QtCore.QRect(10, 430, 531, 41))
        self.button.setObjectName("pushButton")

        self.frame_2 = QtGui.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(550, 10, 531, 571))
        self.frame_2.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")

        self.figure = Figure()
        self.loss = FigureCanvas(self.figure)
        self.loss.setParent(self.frame_2)
        self.loss.resize(252,192)
        # self.figure.setGeometry(QtCore.QRect(10, 10, 252, 192))

        self.rightTransformed = PyuEyeQtView(4, callback_fn=get_np_image, parent=self.frame_2)
        self.rightTransformed.setGeometry(QtCore.QRect(270, 10, 252, 192))


        self.calib_depth = PyuEyeQtView(5, callback_fn=get_calib_prediction, parent=self.frame_2)
        self.calib_depth.setGeometry(QtCore.QRect(10, 210, 511, 351))

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.start_pred)
        self.timer.start()


        MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # self.mono.resize(336,256)
        # self.stereo.resize(336,256)
        # self.rightTransformed.resize(336,256)
        # self.calib_depth.resize(336*2,256*2)
        # self.loss.resize(336,256)

        # mainlayout = QVBoxLayout()
        # self.setLayout(mainlayout)
        # self.cmap = plt.cm.ScalarMappable(cmap='jet')
        # views_layout = QHBoxLayout()
        # views_layout.addWidget(self.left)
        # views_layout.addWidget(self.right)
        # prediction_layout = QHBoxLayout()
        # prediction_layout.addWidget(self.mono)
        # prediction_layout.addWidget(self.stereo)
        # mainlayout.addLayout(views_layout)
        # mainlayout.addLayout(prediction_layout)
        #
        #
        # mainlayout.addWidget(button)
        #
        # calib_layout = QHBoxLayout()
        # calib_layout.addWidget(self.loss)
        # calib_layout.addWidget(self.rightTransformed)
        # calib_depth_layout = QHBoxLayout()
        # calib_depth_layout.addWidget(self.calib_depth)
        #
        # mainlayout.addLayout(calib_layout)
        # mainlayout.addLayout(calib_depth_layout)
        # self.left.resize(336, 256)
        # self.right.resize(336, 256)


        # self.stereo = UncalibratedStereo(self.left, self.right)

        # mainlayout.addWidget(self.stereo)
        # layout.addWidget(view2, 0, 2)
        # window.resize(1200, 450)

        # layout.setRowStretch(2, 1)

    def start_calib(self):
        self.calibrateThread.do_calibrate = True
        self.predictionThread.pause_predict_while_calibrating = True

    def start_pred(self):
        self.predictionThread.do_predict = True


        # return QtGui.QImage(mono_out.data, mono_out.shape[1], mono_out.shape[0], QtGui.QImage.Format_RGB888)


if __name__ == '__main__':

    # mono_net = 'phase-mask'
    mono_net = 'midas'
    # calibTrain = CalibTrain(mono_net='phase-mask')
    calibTrain = CalibTrain(mono_net=mono_net)

    app = PyuEyeQtApp()
    MainWindow = QtGui.QMainWindow()
    win = CalibDisp(calibTrain, MainWindow)
    win.setWindowTitle("ICCV Calibration Demo")
    MainWindow.show()

    # t1, c1 = init_cam(0, [win.left, win.mono])
    t1, c1 = init_cam(0, win.left)
    t2, c2 = init_cam(1, win.right)
    # t3 = win.stereo.start()
    # win.show()

    # view2 = PyuEyeQtView(2)
    # view2.user_callback = get_prediction

    # calibTrain = CalibTrain(mono_net='midas')



    # cleanup
    app.exit_connect(t1.stop)
    app.exit_connect(t2.stop)
    sys.exit(app.exec_())
    # app.exec_()
    # app.exec_()

    win.predictionThread.join()
    win.calibrateThread.join()

    t1.stop()
    t1.join()

    t2.stop()
    t2.join()

    c1.stop_video()
    c1.exit()

    c2.stop_video()
    c2.exit()



if __name__ == "__main__":
    main()

