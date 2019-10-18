#!/usr/bin/env python

#------------------------------------------------------------------------------
#                 PyuEye example - gui application modul
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

# from PyQt4 import QtCore
# from PyQt4 import QtGui
from PyQt5.QtWidgets import QGraphicsScene, QApplication
from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSlider, QWidget
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
import matplotlib.pyplot as plt
import numpy as np

from pyueye import ueye
import time


def get_qt_format(ueye_color_format):
    return { ueye.IS_CM_SENSOR_RAW8: QtGui.QImage.Format_Mono,
             ueye.IS_CM_MONO8: QtGui.QImage.Format_Mono,
             ueye.IS_CM_RGB8_PACKED: QtGui.QImage.Format_RGB888,
             ueye.IS_CM_BGR8_PACKED: QtGui.QImage.Format_RGB888,
             ueye.IS_CM_RGBA8_PACKED: QtGui.QImage.Format_RGB32,
             ueye.IS_CM_BGRA8_PACKED: QtGui.QImage.Format_RGB32
    } [ueye_color_format]



class PyuEyeQtView(QtWidgets.QWidget):

    update_signal = QtCore.pyqtSignal(QtGui.QImage, name='update_signal')

    def __init__(self, cam_id, callback_fn=None, parent=None):
        super(self.__class__, self).__init__(parent)

        self.image = None
        # print(hex(id(self.image)))

        self.graphics_view = QtWidgets.QGraphicsView(self)
        self.v_layout = QtWidgets.QVBoxLayout(self)
        self.h_layout = QtWidgets.QHBoxLayout()

        self.scene = QtWidgets.QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        self.v_layout.addWidget(self.graphics_view)

        self.scene.drawBackground = self.draw_background
        # self.scene.setSceneRect(self.scene.itemsBoundingRect())
        self.update_signal.connect(self.update_image)

        self.processors = []
        #self.resize(640, 512)
        # self.resize(612, 460)

        self.v_layout.addLayout(self.h_layout)
        self.setLayout(self.v_layout)
        self.cam_id = cam_id
        self.call_back_fn = callback_fn if callback_fn else self.user_callback
        self.image_arr = None
        self.update = True
        self.last_update = time.time()

    def on_update_canny_1_slider(self, value):
        pass # print(value)

    def on_update_canny_2_slider(self, value):
        pass # print(value)

    def draw_background(self, painter, rect):
        if self.image:
            # image = self.image.scaled(rect.width(), rect.height(), QtCore.Qt.KeepAspectRatio)
            image = self.image.scaled(rect.width(), rect.height())
            painter.drawImage(rect.x(), rect.y(), image)

    def update_image(self, image):
        self.scene.update()

    def user_callback(self, image_data, cam_id):
        return image_data.as_cv_image()

    def get_np_image(self, image_data):
        cmap = plt.cm.ScalarMappable(cmap='jet')
        image_data = np.transpose(image_data[0], (1, 2, 0))
        image_data = np.clip(image_data, 0, 1)
        image_data = cmap.to_rgba(image_data, bytes=True)[:, :, :3]
        image_data = np.array(image_data, dtype=np.uint8)
        return QtGui.QImage(image_data.data, image_data.shape[1], image_data.shape[0], QtGui.QImage.Format_RGB888)

    def show_right_transformed(self, image_data):
        self.image = self.get_np_image(image_data)
        self.update_signal.emit(self.image)

    def show_sample_image(self, image_data):
        cmap = plt.cm.ScalarMappable(cmap='jet')
        # image_data = np.transpose(image_data, (1, 2, 0))
        # image_data = np.clip(image_data, 0, 1)
        # image_data = cmap.to_rgba(image_data, bytes=True)
        image_data = np.array(image_data, dtype=np.uint8)
        self.image =  QtGui.QImage(image_data.data, image_data.shape[1], image_data.shape[0], QtGui.QImage.Format_RGB888)
        self.update_signal.emit(self.image)

    def handle(self, image_data):
        if self.update:
            self.last_update = time.time()
            self.image = self.call_back_fn(self, image_data, self.cam_id)
            # print(hex(id(self.image)))

            self.update_signal.emit(self.image)

            # unlock the buffer so we can use it again
            try:
                image_data.unlock()
            except:
                pass

    def shutdown(self):
        self.close()

    def add_processor(self, callback):
        self.processors.append(callback)
    

class PyuEyeQtApp:
    def __init__(self, args=[]):        
        self.qt_app = QtWidgets.QApplication(args)
            
    def exec_(self):
        self.qt_app.exec_()

    def exit_connect(self, method):
        self.qt_app.aboutToQuit.connect(method)
