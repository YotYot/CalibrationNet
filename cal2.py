# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'cal.ui'
#
# Created by: PyQt4 UI code generator 4.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1087, 624)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.frame = QtGui.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10, 10, 531, 411))
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.graphicsView = QtGui.QGraphicsView(self.frame)
        self.graphicsView.setGeometry(QtCore.QRect(10, 10, 252, 192))
        self.graphicsView.setObjectName(_fromUtf8("graphicsView"))
        self.graphicsView_2 = QtGui.QGraphicsView(self.frame)
        self.graphicsView_2.setGeometry(QtCore.QRect(270, 10, 252, 192))
        self.graphicsView_2.setObjectName(_fromUtf8("graphicsView_2"))
        self.graphicsView_3 = QtGui.QGraphicsView(self.frame)
        self.graphicsView_3.setGeometry(QtCore.QRect(10, 210, 252, 192))
        self.graphicsView_3.setObjectName(_fromUtf8("graphicsView_3"))
        self.graphicsView_4 = QtGui.QGraphicsView(self.frame)
        self.graphicsView_4.setGeometry(QtCore.QRect(270, 210, 252, 192))
        self.graphicsView_4.setObjectName(_fromUtf8("graphicsView_4"))
        self.pushButton = QtGui.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(10, 430, 531, 41))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.frame_2 = QtGui.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(550, 10, 531, 571))
        self.frame_2.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_2.setObjectName(_fromUtf8("frame_2"))
        self.graphicsView_9 = QtGui.QGraphicsView(self.frame_2)
        self.graphicsView_9.setGeometry(QtCore.QRect(10, 10, 252, 192))
        self.graphicsView_9.setObjectName(_fromUtf8("graphicsView_9"))
        self.graphicsView_10 = QtGui.QGraphicsView(self.frame_2)
        self.graphicsView_10.setGeometry(QtCore.QRect(270, 10, 252, 192))
        self.graphicsView_10.setObjectName(_fromUtf8("graphicsView_10"))
        self.graphicsView_11 = QtGui.QGraphicsView(self.frame_2)
        self.graphicsView_11.setGeometry(QtCore.QRect(10, 210, 511, 351))
        self.graphicsView_11.setObjectName(_fromUtf8("graphicsView_11"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1087, 20))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuICCV = QtGui.QMenu(self.menubar)
        self.menuICCV.setObjectName(_fromUtf8("menuICCV"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuICCV.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.pushButton.setText(_translate("MainWindow", "Calibrate", None))
        self.menuICCV.setTitle(_translate("MainWindow", "ICCV", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.graphicsView
    sys.exit(app.exec_())

