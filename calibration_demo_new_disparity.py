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
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import *
import PyQt5.QtCore
from time import sleep
from threading import Thread
import threading
import re

import torch
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Qt5Agg')

# import demo
from pyueye import ueye
from calib_train_disparity import CalibTrain

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
    # reshape the image data as 1dimensional array
    image = image_data.as_1d_image()
    if cam_id == 0 or cam_id == 1:
        obj.image_arr = image

    # show the image with Qt
    return QtGui.QImage(image.data,image_data.mem_info.width,image_data.mem_info.height,QtGui.QImage.Format_RGB888)


def get_stereo_prediction(obj, image_data, cam_id):

    # reshape the image data as 1dimensional array
    return QtGui.QImage(image_data.data, image_data.shape[1], image_data.shape[0], QtGui.QImage.Format_RGB888)
    # return QtGui.QImage(big_image.data,big_image.shape[1],big_image.shape[0],QtGui.QImage.Format_RGB888)

def get_calib_prediction(obj, image_data, cam_id):

    # reshape the image data as 1dimensional array
    image_data = cmap.to_rgba(image_data[0], bytes=True)[:, :, :3]
    image_data = np.array(image_data, dtype=np.uint8)
    return QtGui.QImage(image_data.data, image_data.shape[1], image_data.shape[0], QtGui.QImage.Format_RGB888)

def get_np_image(obj, image_data, cam_id):

    # reshape the image data as 1dimensional array
    image_data = np.transpose(image_data[0], (1,2,0))
    image_data = np.clip(image_data,0,1)
    image_data = cmap.to_rgba(image_data, bytes=True)[:, :, :3]
    image_data = np.array(image_data, dtype=np.uint8)
    return QtGui.QImage(image_data.data, image_data.shape[1], image_data.shape[0], QtGui.QImage.Format_RGB888)


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
            stereo_unrect, mono_out  = win.calibTrain.get_mono_prediction(win.left.image_arr, win.right.image_arr)
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
        global c1,c2, t1, t2
        self.win.left.update = False
        self.win.right.update = False
        # c1.freeze_video()
        # c2.freeze_video()
        c1.exit()
        c2.exit()
        self.calibrate()
        if self.win.livefeed:
            t1.pause = True
            t2.pause = True
            time.sleep(1)
            t2, c2 = init_cam(right_cam_idx, win.right)
            t1, c1 = init_cam(left_cam_idx, win.left)
            c1.capture_video()
            c2.capture_video()
        calibTrain = CalibTrain(mono_net=self.win.mononet_str, lr = self.win.lr)
        self.calibTrain = calibTrain
        self.calibTrain.scheduled_lr = self.win.scheduled_lr
        self.win.calibTrain = calibTrain
        self.win.left.update = True
        self.win.right.update = True
        self.win.predictionThread.pause_predict_while_calibrating = False

    def calibrate(self):
        self.calibTrain.best_test_loss = 200.0
        self.win.figure.clf()
        # if win.livefeed:
        self.calibTrain.calibrate(obj=win, l_img=win.left.image_arr, r_img=win.right.image_arr, epoch_num=win.epoch_num)
        # else:
        #     self.calibTrain.calibrate(obj=win)


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtWidgets.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig)

class CalibDisp(QtWidgets.QWidget):
    def __init__(self, calibTrain, Form):
        super(CalibDisp, self).__init__()

        Form.setObjectName("Form")
        Form.resize(907, 553)
        Form.setMinimumSize(QtCore.QSize(698, 0))
        self.scheduled_lr = False
        self.lr = 0.0001
        self.epoch_num = 10
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.StaticPart = QtWidgets.QGroupBox(Form)
        self.StaticPart.setObjectName("StaticPart")
        self.gridLayout = QtWidgets.QGridLayout(self.StaticPart)
        self.gridLayout.setObjectName("gridLayout")

        self.left = PyuEyeQtView(cam_id=0, callback_fn=process_image, parent=self.StaticPart)
        self.left.setObjectName("Left")
        self.gridLayout.addWidget(self.left, 1, 0, 1, 1)

        self.LeftLabel = QtWidgets.QLabel(self.StaticPart)
        self.LeftLabel.setObjectName("LeftLabel")
        self.gridLayout.addWidget(self.LeftLabel, 0, 0, 1, 1)

        self.right = PyuEyeQtView(cam_id=1, callback_fn=process_image, parent=self.StaticPart)
        self.right.setObjectName("Right")
        self.gridLayout.addWidget(self.right, 1, 1, 1, 1)

        self.RightLabel = QtWidgets.QLabel(self.StaticPart)
        self.RightLabel.setObjectName("RightLabel")
        self.gridLayout.addWidget(self.RightLabel, 0, 1, 1, 1)

        self.mono = PyuEyeQtView(2, parent=self.StaticPart)
        self.mono.setObjectName("Mono")
        self.gridLayout.addWidget(self.mono, 3, 1, 1, 1)
        self.MonoLabel = QtWidgets.QLabel(self.StaticPart)
        self.MonoLabel.setObjectName("MonoLabel")
        self.gridLayout.addWidget(self.MonoLabel, 2, 1, 1, 1)

        self.stereo = PyuEyeQtView(3, callback_fn=get_stereo_prediction, parent=self.StaticPart)
        self.stereo.setObjectName("Stereo")
        self.gridLayout.addWidget(self.stereo, 3, 0, 1, 1)

        self.StereoLabel = QtWidgets.QLabel(self.StaticPart)
        self.StereoLabel.setObjectName("StereoLabel")
        self.gridLayout.addWidget(self.StereoLabel, 2, 0, 1, 1)

        self.MonoNet = QtWidgets.QCheckBox(self.StaticPart)
        self.MonoNet.setObjectName("MonoNet")
        if mono_net == 'midas':
            self.MonoNet.setChecked(True)
        else:
            self.MonoNet.setChecked(False)
        self.mononet = self.MonoNet.isChecked()
        self.mononet_str = 'phase-mask' if not self.mononet else 'midas'
        self.MonoNet.stateChanged.connect(self.mononet_changed)
        self.gridLayout.addWidget(self.MonoNet, 4, 1, 1, 1)

        self.LiveFeed = QtWidgets.QCheckBox(self.StaticPart)
        self.LiveFeed.setChecked(True)
        self.LiveFeed.setObjectName("LiveFeed")
        self.livefeed = self.LiveFeed.isChecked()
        self.LiveFeed.stateChanged.connect(self.livefeed_changed)
        self.gridLayout.addWidget(self.LiveFeed, 4, 0, 1, 1)

        self.EpochNumLabel = QtWidgets.QLabel(self.StaticPart)
        self.EpochNumLabel.setObjectName("label")
        self.gridLayout.addWidget(self.EpochNumLabel, 5, 0, 1, 1)

        self.EpochNum = QtWidgets.QComboBox(self.StaticPart)
        self.EpochNum.setObjectName("comboBox")
        self.EpochNum.addItem("")
        self.EpochNum.addItem("")
        self.EpochNum.addItem("")
        self.EpochNum.addItem("")
        self.EpochNum.addItem("")
        self.EpochNum.currentIndexChanged.connect(self.update_epoch_num)
        self.gridLayout.addWidget(self.EpochNum, 6, 0, 1, 1)

        self.LearningRateLabel = QtWidgets.QLabel(self.StaticPart)
        self.LearningRateLabel.setObjectName("label")
        self.gridLayout.addWidget(self.LearningRateLabel, 5, 1, 1, 1)

        self.LearningRate = QtWidgets.QComboBox(self.StaticPart)
        self.LearningRate.setObjectName("LR_ComboBox")
        self.LearningRate.addItem("")
        self.LearningRate.addItem("")
        self.LearningRate.addItem("")
        self.LearningRate.addItem("")
        self.LearningRate.addItem("")
        self.LearningRate.addItem("")
        self.LearningRate.addItem("")
        if mono_net == 'midas':
            self.LearningRate.setCurrentIndex(3)
        else:
            self.LearningRate.setCurrentIndex(2)
        self.LearningRate.currentIndexChanged.connect(self.update_lr)
        self.gridLayout.addWidget(self.LearningRate, 6, 1, 1, 1)

        self.CalibrateButton = QtWidgets.QPushButton(self.StaticPart)
        self.CalibrateButton.setObjectName("CalibrateButton")
        self.CalibrateButton.clicked.connect(self.start_calib)
        self.gridLayout.addWidget(self.CalibrateButton, 7, 0, 1, 2)

        self.horizontalLayout.addWidget(self.StaticPart)

        self.CalibratioPart = QtWidgets.QGroupBox(Form)
        self.CalibratioPart.setObjectName("CalibratioPart")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.CalibratioPart)
        self.gridLayout_3.setObjectName("gridLayout_3")

        self.calib_depth = PyuEyeQtView(5, callback_fn=get_calib_prediction)
        self.calib_depth.setObjectName("Stereo_calibrated")
        self.gridLayout_3.addWidget(self.calib_depth, 3, 0, 1, 1)

        self.figure = Figure(figsize=(5, 4))
        self.loss = FigureCanvas(self.figure)
        self.loss.setObjectName("Loss")
        self.gridLayout_3.addWidget(self.loss, 1, 0, 1, 1)

        self.LossLabel = QtWidgets.QLabel(self.CalibratioPart)
        self.LossLabel.setObjectName("LossLabel")
        self.gridLayout_3.addWidget(self.LossLabel, 0, 0, 1, 1)

        self.StereoAfterLabel = QtWidgets.QLabel(self.CalibratioPart)
        self.StereoAfterLabel.setObjectName("StereoAfterLabel")
        self.gridLayout_3.addWidget(self.StereoAfterLabel, 2, 0, 1, 1)
        self.horizontalLayout.addWidget(self.CalibratioPart)
        self.CalibratioPart.raise_()

        self.calibTrain = calibTrain
        self.calibrateThread = CalibrateThread(win=self, calibTrain=self.calibTrain)
        self.calibrateThread.start()
        #
        self.predictionThread = MonoStereoPredictionThread(win=self)
        self.predictionThread.start()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.start_pred)
        self.timer.start()


        # Form.setCentralWidget(self.centralwidget)
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def update_epoch_num(self):
        self.epoch_num = int(self.EpochNum.currentText())

    def update_lr(self):
        lr = re.sub("[^0-9.]", "", self.LearningRate.currentText())
        if 'Sch' in self.LearningRate.currentText():
            self.calibTrain.scheduled_lr = True
            self.scheduled_lr = True

        else:
            self.calibTrain.scheduled_lr = False
            self.scheduled_lr = False
        print ("Scheduled: {}, LR: {}".format(self.calibTrain.scheduled_lr, lr))
        self.lr             = float(lr)
        self.calibTrain.lr  = self.lr
        self.calibTrain.optimizer = self.calibTrain.get_optimizer(self.calibTrain.calibration_model)

    def mononet_changed(self, int):
        self.mononet = self.MonoNet.isChecked()
        self.mononet_str = 'phase-mask' if not self.mononet else 'midas'
        calibTrain = CalibTrain(mono_net=self.mononet_str, lr=self.lr, scheduled_lr=self.scheduled_lr)
        self.calibTrain = calibTrain

    def livefeed_changed(self, int):
        global c1,c2, t1, t2
        self.livefeed = self.LiveFeed.isChecked()
        if not self.livefeed:
            # c1.freeze_video()
            # c2.freeze_video()
            c1.exit()
            c2.exit()
            time.sleep(2)
            self.left.update = False
            self.right.update = False
            l_sample_image = plt.imread(os.path.join('Sample_Images', 'L_10.tif'))
            r_sample_image = plt.imread(os.path.join('Sample_Images', 'R_10.tif'))
            self.left.show_sample_image(l_sample_image)
            self.right.show_sample_image(r_sample_image)
            self.left.image_arr = l_sample_image
            self.right.image_arr = r_sample_image
        else:
            self.left.update = True
            self.right.update = True
            t1.pause = True
            t2.pause = True
            time.sleep(1)
            t2, c2 = init_cam(right_cam_idx, win.right)
            t1, c1 = init_cam(left_cam_idx, win.left)
            c1.capture_video()
            c2.capture_video()


    def start_calib(self):
        self.calibrateThread.do_calibrate = True
        self.predictionThread.pause_predict_while_calibrating = True

    def start_pred(self):
        self.predictionThread.do_predict = True

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.LeftLabel.setText(_translate("Form", "Left"))
        self.MonoLabel.setText(_translate("Form", "Mono Depth"))
        self.LiveFeed.setText(_translate("Form", "Use live feed images"))
        self.CalibrateButton.setText(_translate("Form", "Calibrate!"))
        self.RightLabel.setText(_translate("Form", "Right (transformed)"))
        self.StereoLabel.setText(_translate("Form", "Stereo (B. Calib.)"))
        self.MonoNet.setText(_translate("Form", "Use image based mono-method"))
        self.EpochNumLabel.setText(_translate("Form", "Epochs:"))
        self.EpochNum.setItemText(0, _translate("Form", "10"))
        self.EpochNum.setItemText(1, _translate("Form", "20"))
        self.EpochNum.setItemText(2, _translate("Form", "50"))
        self.EpochNum.setItemText(3, _translate("Form", "100"))
        self.EpochNum.setItemText(4, _translate("Form", "200"))
        self.LearningRateLabel.setText(_translate("Form", "Learning Rate:"))
        self.LearningRate.setItemText(0, _translate("Form", "0.1"))
        self.LearningRate.setItemText(1, _translate("Form", "0.01"))
        self.LearningRate.setItemText(2, _translate("Form", "0.001"))
        self.LearningRate.setItemText(3, _translate("Form", "0.0001"))
        self.LearningRate.setItemText(4, _translate("Form", "0.00001"))
        self.LearningRate.setItemText(5, _translate("Form", "Scheduled, 0.001"))
        self.LearningRate.setItemText(6, _translate("Form", "Scheduled, 0.0001"))
        self.LossLabel.setText(_translate("Form", "Depth Maps Consistency Loss"))
        self.StereoAfterLabel.setText(_translate("Form", "Stereo Depth After Calibration"))
        # return QtGui.QImage(mono_out.data, mono_out.shape[1], mono_out.shape[0], QtGui.QImage.Format_RGB888)


if __name__ == '__main__':
    left_cam_idx = 0
    right_cam_idx = 1

    mono_net = 'phase-mask'
    # mono_net = 'midas'
    # calibTrain = CalibTrain(mono_net='phase-mask')
    lr = 0.0001 if (mono_net == 'midas') else 0.001
    calibTrain = CalibTrain(mono_net=mono_net, lr = lr)

    app = PyuEyeQtApp()
    # MainWindow = QtWidgets.QMainWindow()
    Form = QtWidgets.QWidget()
    win = CalibDisp(calibTrain, Form)
    win.setWindowTitle("ICCV Calibration Demo")
    Form.show()

    # t1, c1 = init_cam(0, [win.left, win.mono])
    t2, c2 = init_cam(right_cam_idx, win.right)
    t1, c1 = init_cam(left_cam_idx, win.left)

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

