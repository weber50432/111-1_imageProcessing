# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'process.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1567, 957)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(1567, 957))
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(41, 42, 1481, 891))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.loadButton = QtWidgets.QPushButton(self.layoutWidget)
        self.loadButton.setObjectName("loadButton")
        self.verticalLayout_2.addWidget(self.loadButton)
        self.autoAdjustment = QtWidgets.QPushButton(self.layoutWidget)
        self.autoAdjustment.setObjectName("autoAdjustment")
        self.verticalLayout_2.addWidget(self.autoAdjustment)
        self.confirmButton = QtWidgets.QPushButton(self.layoutWidget)
        self.confirmButton.setObjectName("confirmButton")
        self.verticalLayout_2.addWidget(self.confirmButton)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_3.addWidget(self.label_4)
        self.label_5 = QtWidgets.QLabel(self.layoutWidget)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_3.addWidget(self.label_5)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.spatialResolution = QtWidgets.QLineEdit(self.layoutWidget)
        self.spatialResolution.setObjectName("spatialResolution")
        self.verticalLayout_4.addWidget(self.spatialResolution)
        self.grayscaleLevel = QtWidgets.QLineEdit(self.layoutWidget)
        self.grayscaleLevel.setObjectName("grayscaleLevel")
        self.verticalLayout_4.addWidget(self.grayscaleLevel)
        self.horizontalLayout.addLayout(self.verticalLayout_4)
        self.BC = QtWidgets.QVBoxLayout()
        self.BC.setObjectName("BC")
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setObjectName("label_2")
        self.BC.addWidget(self.label_2)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        self.label_3.setObjectName("label_3")
        self.BC.addWidget(self.label_3)
        self.horizontalLayout.addLayout(self.BC)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.spinBox_2 = QtWidgets.QSpinBox(self.layoutWidget)
        self.spinBox_2.setObjectName("spinBox_2")
        self.verticalLayout_5.addWidget(self.spinBox_2)
        self.spinBox = QtWidgets.QSpinBox(self.layoutWidget)
        self.spinBox.setObjectName("spinBox")
        self.verticalLayout_5.addWidget(self.spinBox)
        self.horizontalLayout.addLayout(self.verticalLayout_5)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.BrightnessSlider = QtWidgets.QSlider(self.layoutWidget)
        self.BrightnessSlider.setMinimum(-99)
        self.BrightnessSlider.setOrientation(QtCore.Qt.Horizontal)
        self.BrightnessSlider.setObjectName("BrightnessSlider")
        self.verticalLayout.addWidget(self.BrightnessSlider)
        self.contrastSlider = QtWidgets.QSlider(self.layoutWidget)
        self.contrastSlider.setSliderPosition(20)
        self.contrastSlider.setOrientation(QtCore.Qt.Horizontal)
        self.contrastSlider.setObjectName("contrastSlider")
        self.verticalLayout.addWidget(self.contrastSlider)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_6.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.original = QtWidgets.QLabel(self.layoutWidget)
        self.original.setText("")
        self.original.setObjectName("original")
        self.horizontalLayout_2.addWidget(self.original)
        self.gray1 = QtWidgets.QLabel(self.layoutWidget)
        self.gray1.setText("")
        self.gray1.setObjectName("gray1")
        self.horizontalLayout_2.addWidget(self.gray1)
        self.histogram1 = QtWidgets.QOpenGLWidget(self.layoutWidget)
        self.histogram1.setObjectName("histogram1")
        self.horizontalLayout_2.addWidget(self.histogram1)
        self.verticalLayout_6.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.compare = QtWidgets.QLabel(self.layoutWidget)
        self.compare.setText("")
        self.compare.setObjectName("compare")
        self.horizontalLayout_3.addWidget(self.compare)
        self.gray2 = QtWidgets.QLabel(self.layoutWidget)
        self.gray2.setText("")
        self.gray2.setObjectName("gray2")
        self.horizontalLayout_3.addWidget(self.gray2)
        self.histogram2 = QtWidgets.QLabel(self.layoutWidget)
        self.histogram2.setText("")
        self.histogram2.setObjectName("histogram2")
        self.horizontalLayout_3.addWidget(self.histogram2)
        self.verticalLayout_6.addLayout(self.horizontalLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.loadButton.setText(_translate("MainWindow", "Load"))
        self.autoAdjustment.setText(_translate("MainWindow", "Auto adjustment"))
        self.confirmButton.setText(_translate("MainWindow", "confirm"))
        self.label_4.setText(_translate("MainWindow", "spatial resolution"))
        self.label_5.setText(_translate("MainWindow", "Grayscale levels"))
        self.label_2.setText(_translate("MainWindow", "Brightness"))
        self.label_3.setText(_translate("MainWindow", "contrast"))
