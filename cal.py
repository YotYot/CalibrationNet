# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'cal.ui'
#
# Created: Wed Sep  4 15:59:38 2019
#      by: PyQt4 UI code generator 4.6.2
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(553, 1100)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtGui.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10, 10, 531, 411))
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.graphicsView = QtGui.QGraphicsView(self.frame)
        self.graphicsView.setGeometry(QtCore.QRect(10, 10, 252, 192))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QtGui.QGraphicsView(self.frame)
        self.graphicsView_2.setGeometry(QtCore.QRect(270, 10, 252, 192))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.graphicsView_3 = QtGui.QGraphicsView(self.frame)
        self.graphicsView_3.setGeometry(QtCore.QRect(10, 210, 252, 192))
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.graphicsView_4 = QtGui.QGraphicsView(self.frame)
        self.graphicsView_4.setGeometry(QtCore.QRect(270, 210, 252, 192))
        self.graphicsView_4.setObjectName("graphicsView_4")
        self.pushButton = QtGui.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(10, 430, 531, 27))
        self.pushButton.setObjectName("pushButton")
        self.frame_2 = QtGui.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(10, 460, 531, 571))
        self.frame_2.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.graphicsView_9 = QtGui.QGraphicsView(self.frame_2)
        self.graphicsView_9.setGeometry(QtCore.QRect(10, 10, 252, 192))
        self.graphicsView_9.setObjectName("graphicsView_9")
        self.graphicsView_10 = QtGui.QGraphicsView(self.frame_2)
        self.graphicsView_10.setGeometry(QtCore.QRect(270, 10, 252, 192))
        self.graphicsView_10.setObjectName("graphicsView_10")
        self.graphicsView_11 = QtGui.QGraphicsView(self.frame_2)
        self.graphicsView_11.setGeometry(QtCore.QRect(10, 210, 511, 351))
        self.graphicsView_11.setObjectName("graphicsView_11")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 553, 27))
        self.menubar.setObjectName("menubar")
        self.menuICCV = QtGui.QMenu(self.menubar)
        self.menuICCV.setObjectName("menuICCV")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuICCV.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton.setText(QtGui.QApplication.translate("MainWindow", "Calibrate", None, QtGui.QApplication.UnicodeUTF8))
        self.menuICCV.setTitle(QtGui.QApplication.translate("MainWindow", "ICCV", None, QtGui.QApplication.UnicodeUTF8))

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())