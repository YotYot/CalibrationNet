# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/yotamg/Desktop/untitled.ui'
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
        MainWindow.resize(943, 616)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))

        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.graphicsView = QtGui.QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName(_fromUtf8("graphicsView"))
        self.verticalLayout_3.addWidget(self.graphicsView)
        self.graphicsView_2 = QtGui.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setObjectName(_fromUtf8("graphicsView_2"))
        self.verticalLayout_3.addWidget(self.graphicsView_2)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)

        self.verticalLayout_4 = QtGui.QVBoxLayout()
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.graphicsView_3 = QtGui.QGraphicsView(self.centralwidget)
        self.graphicsView_3.setObjectName(_fromUtf8("graphicsView_3"))
        self.verticalLayout_4.addWidget(self.graphicsView_3)
        self.graphicsView_4 = QtGui.QGraphicsView(self.centralwidget)
        self.graphicsView_4.setObjectName(_fromUtf8("graphicsView_4"))
        self.verticalLayout_4.addWidget(self.graphicsView_4)
        self.horizontalLayout_2.addLayout(self.verticalLayout_4)


        self.pushButton = QtGui.QPushButton(self.centralwidget)
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.horizontalLayout_2.addWidget(self.pushButton)

        self.verticalLayout_5 = QtGui.QVBoxLayout()
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))

        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.graphicsView_5 = QtGui.QGraphicsView(self.centralwidget)
        self.graphicsView_5.setObjectName(_fromUtf8("graphicsView_5"))
        self.graphicsView_6 = QtGui.QGraphicsView(self.centralwidget)
        self.graphicsView_6.setObjectName(_fromUtf8("graphicsView_6"))
        self.horizontalLayout_3.addWidget(self.graphicsView_5)
        self.horizontalLayout_3.addWidget(self.graphicsView_6)

        self.graphicsView_7 = QtGui.QGraphicsView(self.centralwidget)
        self.graphicsView_7.setObjectName(_fromUtf8("graphicsView_7"))

        self.verticalLayout_5.addLayout(self.horizontalLayout_3)
        self.verticalLayout_5.addWidget(self.graphicsView_7)

        self.horizontalLayout_2.addLayout(self.verticalLayout_5)



        # self.horizontalLayout = QtGui.QHBoxLayout()
        # self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        # self.graphicsView_6 = QtGui.QGraphicsView(self.centralwidget)
        # self.graphicsView_6.setObjectName(_fromUtf8("graphicsView_6"))
        # self.horizontalLayout.addWidget(self.graphicsView_6)
        # self.graphicsView_5 = QtGui.QGraphicsView(self.centralwidget)
        # self.graphicsView_5.setObjectName(_fromUtf8("graphicsView_5"))
        # self.horizontalLayout.addWidget(self.graphicsView_5)
        # self.horizontalLayout_2.addLayout(self.horizontalLayout)
        # self.graphicsView_7 = QtGui.QGraphicsView(self.centralwidget)
        # self.graphicsView_7.setObjectName(_fromUtf8("graphicsView_7"))
        # self.horizontalLayout_2.addWidget(self.graphicsView_7)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.pushButton.setText(_translate("MainWindow", "Calibrate!", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

