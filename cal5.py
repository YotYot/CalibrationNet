# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/yotamg/Desktop/aaa.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(907, 553)
        Form.setMinimumSize(QtCore.QSize(698, 0))
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.StaticPart = QtWidgets.QGroupBox(Form)
        self.StaticPart.setObjectName("StaticPart")
        self.gridLayout = QtWidgets.QGridLayout(self.StaticPart)
        self.gridLayout.setObjectName("gridLayout")
        self.Mono = QtWidgets.QGraphicsView(self.StaticPart)
        self.Mono.setObjectName("Mono")
        self.gridLayout.addWidget(self.Mono, 3, 0, 1, 1)
        self.LeftLabel = QtWidgets.QLabel(self.StaticPart)
        self.LeftLabel.setObjectName("LeftLabel")
        self.gridLayout.addWidget(self.LeftLabel, 0, 0, 1, 1)
        self.Stereo = QtWidgets.QGraphicsView(self.StaticPart)
        self.Stereo.setObjectName("Stereo")
        self.gridLayout.addWidget(self.Stereo, 3, 1, 1, 1)
        self.Left = QtWidgets.QGraphicsView(self.StaticPart)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Left.sizePolicy().hasHeightForWidth())
        self.Left.setSizePolicy(sizePolicy)
        self.Left.setObjectName("Left")
        self.gridLayout.addWidget(self.Left, 1, 0, 1, 1)
        self.Right = QtWidgets.QGraphicsView(self.StaticPart)
        self.Right.setEnabled(True)
        self.Right.setObjectName("Right")
        self.gridLayout.addWidget(self.Right, 1, 1, 1, 1)
        self.MonoLabel = QtWidgets.QLabel(self.StaticPart)
        self.MonoLabel.setObjectName("MonoLabel")
        self.gridLayout.addWidget(self.MonoLabel, 2, 0, 1, 1)
        self.LiveFeed = QtWidgets.QCheckBox(self.StaticPart)
        self.LiveFeed.setChecked(True)
        self.LiveFeed.setObjectName("LiveFeed")
        self.gridLayout.addWidget(self.LiveFeed, 4, 0, 1, 1)
        self.CalibrateButton = QtWidgets.QPushButton(self.StaticPart)
        self.CalibrateButton.setObjectName("CalibrateButton")
        self.gridLayout.addWidget(self.CalibrateButton, 9, 0, 1, 2)
        self.RightLabel = QtWidgets.QLabel(self.StaticPart)
        self.RightLabel.setObjectName("RightLabel")
        self.gridLayout.addWidget(self.RightLabel, 0, 1, 1, 1)
        self.StereoLabel = QtWidgets.QLabel(self.StaticPart)
        self.StereoLabel.setObjectName("StereoLabel")
        self.gridLayout.addWidget(self.StereoLabel, 2, 1, 1, 1)
        self.MonoNet = QtWidgets.QCheckBox(self.StaticPart)
        self.MonoNet.setEnabled(True)
        self.MonoNet.setChecked(True)
        self.MonoNet.setObjectName("MonoNet")
        self.gridLayout.addWidget(self.MonoNet, 5, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.StaticPart)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 4, 1, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.StaticPart)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.currentIndexChanged.connect(self.paaa)
        self.gridLayout.addWidget(self.comboBox, 5, 1, 1, 1)
        self.horizontalLayout.addWidget(self.StaticPart)
        self.CalibratioPart = QtWidgets.QGroupBox(Form)
        self.CalibratioPart.setObjectName("CalibratioPart")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.CalibratioPart)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.Stereo_calibrated = QtWidgets.QGraphicsView(self.CalibratioPart)
        self.Stereo_calibrated.setObjectName("Stereo_calibrated")
        self.gridLayout_3.addWidget(self.Stereo_calibrated, 3, 0, 1, 1)
        self.Loss = QtWidgets.QGraphicsView(self.CalibratioPart)
        self.Loss.setEnabled(True)
        self.Loss.setObjectName("Loss")
        self.gridLayout_3.addWidget(self.Loss, 1, 0, 1, 1)
        self.LossLabel = QtWidgets.QLabel(self.CalibratioPart)
        self.LossLabel.setObjectName("LossLabel")
        self.gridLayout_3.addWidget(self.LossLabel, 0, 0, 1, 1)
        self.StereoAfterLabel = QtWidgets.QLabel(self.CalibratioPart)
        self.StereoAfterLabel.setObjectName("StereoAfterLabel")
        self.gridLayout_3.addWidget(self.StereoAfterLabel, 2, 0, 1, 1)
        self.horizontalLayout.addWidget(self.CalibratioPart)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def paaa(self):
        print (self.comboBox.currentText())

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.LeftLabel.setText(_translate("Form", "Left"))
        self.MonoLabel.setText(_translate("Form", "Mono Depth"))
        self.LiveFeed.setText(_translate("Form", "Use live feed images"))
        self.CalibrateButton.setText(_translate("Form", "Calibrate!"))
        self.RightLabel.setText(_translate("Form", "Right (transformed)"))
        self.StereoLabel.setText(_translate("Form", "Stereo Depth Before Calibration"))
        self.MonoNet.setText(_translate("Form", "Use image based mono-method"))
        self.label.setText(_translate("Form", "Epochs:"))
        self.comboBox.setItemText(0, _translate("Form", "10"))
        self.comboBox.setItemText(1, _translate("Form", "20"))
        self.comboBox.setItemText(2, _translate("Form", "50"))
        self.comboBox.setItemText(3, _translate("Form", "100"))
        self.LossLabel.setText(_translate("Form", "Depth Maps Consistency Loss"))
        self.StereoAfterLabel.setText(_translate("Form", "Stereo Depth After Calibration"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

