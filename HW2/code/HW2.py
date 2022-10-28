# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'process.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
import sys
import cv2
import imutils
import numpy as np
import pyqtgraph as pg


def calculate_histogram_list(imgA, imgB):
    listA = [0] * 256
    listB = [0] * 256

    for i in range(len(imgA)):
        for j in range(len(imgA[0])):
            for num in range(256):
                if imgA[i][j] == num:
                    listA[num] += 1
                    break

    for i in range(len(imgA)):
        for j in range(len(imgA[0])):
            for num in range(256):
                if imgB[i][j] == num:
                    listB[num] += 1
                    break

    return listA, listB


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1567, 957)
        self.brightness_value_now = 0  # Updated self.brightness_value_now value
        self.contrast_value_now = 0  # Updated self.brightness_value_now value
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(1567, 957))
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(40, 40, 1481, 891))
        self.widget.setObjectName("widget")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.loadButton = QtWidgets.QPushButton(self.widget)
        self.loadButton.setObjectName("loadButton")
        self.verticalLayout_2.addWidget(self.loadButton)
        self.autoAdjustment = QtWidgets.QPushButton(self.widget)
        self.autoAdjustment.setObjectName("autoAdjustment")
        self.verticalLayout_2.addWidget(self.autoAdjustment)
        self.confirmButton = QtWidgets.QPushButton(self.widget)
        self.confirmButton.setObjectName("confirmButton")
        self.verticalLayout_2.addWidget(self.confirmButton)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.thresholdLabel = QtWidgets.QLabel(self.widget)
        self.thresholdLabel.setObjectName("thresholdLabel")
        self.verticalLayout_3.addWidget(self.thresholdLabel)
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_3.addWidget(self.label_4)
        self.label_5 = QtWidgets.QLabel(self.widget)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_3.addWidget(self.label_5)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.threshold = QtWidgets.QLineEdit(self.widget)
        self.threshold.setObjectName("threshold")
        self.verticalLayout_4.addWidget(self.threshold)
        self.spatialResolution = QtWidgets.QLineEdit(self.widget)
        self.spatialResolution.setObjectName("spatialResolution")
        self.verticalLayout_4.addWidget(self.spatialResolution)
        self.grayscaleLevel = QtWidgets.QLineEdit(self.widget)
        self.grayscaleLevel.setObjectName("grayscaleLevel")
        self.verticalLayout_4.addWidget(self.grayscaleLevel)
        self.horizontalLayout.addLayout(self.verticalLayout_4)
        self.BC = QtWidgets.QVBoxLayout()
        self.BC.setObjectName("BC")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.BC.addWidget(self.label_2)
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setObjectName("label_3")
        self.BC.addWidget(self.label_3)
        self.horizontalLayout.addLayout(self.BC)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        # self.spinBox_2 = QtWidgets.QSpinBox(self.widget)
        # self.spinBox_2.setObjectName("spinBox_2")
        # self.verticalLayout.addWidget(self.spinBox_2)
        # self.spinBox = QtWidgets.QSpinBox(self.widget)
        # self.spinBox.setObjectName("spinBox")
        # self.verticalLayout.addWidget(self.spinBox)
        # self.horizontalLayout.addLayout(self.verticalLayout)
        # self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.BrightnessSlider = QtWidgets.QSlider(self.widget)
        self.BrightnessSlider.setMinimum(-99)
        self.BrightnessSlider.setSliderPosition(self.brightness_value_now)
        self.BrightnessSlider.setOrientation(QtCore.Qt.Horizontal)
        self.BrightnessSlider.setObjectName("BrightnessSlider")
        self.verticalLayout.addWidget(self.BrightnessSlider)
        # connect(self.spinBox,SIGNAL(valueChanged(int)),self.BrightnessSlider,SLOT(setValue(int)))
        self.contrastSlider = QtWidgets.QSlider(self.widget)
        self.contrastSlider.setMinimum(-99)
        self.contrastSlider.setSliderPosition(self.contrast_value_now)
        self.contrastSlider.setOrientation(QtCore.Qt.Horizontal)
        self.contrastSlider.setObjectName("contrastSlider")
        self.verticalLayout.addWidget(self.contrastSlider)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_5.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.original = QtWidgets.QLabel(self.widget)
        self.original.setText("")
        self.original.setObjectName("original")
        self.horizontalLayout_2.addWidget(self.original)
        self.gray1 = QtWidgets.QLabel(self.widget)
        self.gray1.setText("")
        self.gray1.setObjectName("gray1")
        self.horizontalLayout_2.addWidget(self.gray1)
        self.verticalLayout_5.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.compare = QtWidgets.QLabel(self.widget)
        self.compare.setText("")
        self.compare.setObjectName("compare")
        self.horizontalLayout_3.addWidget(self.compare)
        self.gray2 = QtWidgets.QLabel(self.widget)
        self.gray2.setText("")
        self.gray2.setObjectName("gray2")
        self.horizontalLayout_3.addWidget(self.gray2)
        self.verticalLayout_5.addLayout(self.horizontalLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.loadButton.clicked.connect(self.loadImage)
        self.autoAdjustment.clicked.connect(self.histogramEqualization)
        self.confirmButton.clicked.connect(self.modify_image)
        self.BrightnessSlider.valueChanged['int'].connect(
            self.brightness_value)
        self.contrastSlider.valueChanged['int'].connect(
            self.contrast_value)
        self.tmp = None  # Will hold the temporary image for display

    def loadImage(self):
        """ This function will load the user selected image
            and set it to label using the setPhoto function
        """
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        self.image = cv2.imread(self.filename, -1)
        self.setPhoto(self.image)

    def setPhoto(self, image):
        """ This function will take image input and resize it 
            only for display purpose and convert it to QImage
            to set at the label.
        """
        self.tmp = image
        image = imutils.resize(image, width=128)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.imgA = np.zeros((len(frame), len(frame[0])))
        self.imgB = np.zeros((len(frame), len(frame[0])))
        for i in range(len(frame)):
            for j in range(len(frame[0])):
                self.imgA[i][j] = int(
                    (int(frame[i][j][0]) + int(frame[i][j][1]) + int(frame[i][j][2])) / 3)
                self.imgB[i][j] = int(int(
                    0.299*frame[i][j][0]) + int(0.587*frame[i][j][1]) + int(0.114*frame[i][j][2]))
        self.imgA = self.imgA.astype(np.uint8)
        self.imgB = self.imgB.astype(np.uint8)
        image = QImage(frame, frame.shape[1], frame.shape[0],
                       frame.strides[0], QImage.Format_RGB888)
        self.original.setPixmap(QtGui.QPixmap.fromImage(image))
        self.showGrayImage(self.imgA, self.imgB)
        self.showBarChart(self.imgA, self.imgB)

    def showGrayImage(self, imgA, imgB):
        height, width = imgA.shape
        qcpImg = QtGui.QImage(self.compare_image(
            imgA, imgB), width, height, width, QtGui.QImage.Format_Indexed8)
        qImgA = QtGui.QImage(imgA.astype(np.uint8), width,
                             height, width, QtGui.QImage.Format_Indexed8)
        qImgB = QtGui.QImage(imgB.astype(np.uint8), width,
                             height, width, QtGui.QImage.Format_Indexed8)
        self.compare.setPixmap(QtGui.QPixmap(qcpImg))
        self.gray1.setPixmap(QtGui.QPixmap(qImgA))
        self.gray2.setPixmap(QtGui.QPixmap(qImgB))

    def compare_image(self, imgA, imgB):
        img = np.zeros((len(imgA), len(imgA[0])))
        for i in range(len(imgA)):
            for j in range(len(imgA[0])):
                img[i][j] = int(imgA[i][j]) - int(imgB[i][j])
                if img[i][j] < 0:
                    img[i][j] = 0
                elif img[i][j] > 255:
                    img[i][j] = 255
        return img.astype(np.uint8)

    def showBarChart(self, imgA, imgB):
        listA, listB = calculate_histogram_list(imgA, imgB)
        x = list(range(256))

        plotA = pg.plot()
        histogram = pg.BarGraphItem(x=x, height=listA, width=0.05)
        plotA.addItem(histogram)

        plotB = pg.plot()
        histogram = pg.BarGraphItem(x=x, height=listB, width=0.05)
        plotB.addItem(histogram)

        self.horizontalLayout_2.addWidget(plotA)
        self.horizontalLayout_3.addWidget(plotB)

    def brightness_value(self, value1):
        self.brightness_value_now = value1

    def contrast_value(self, value2):
        self.contrast_value_now = value2
        self.BrightnessSlider.setSliderPosition(self.brightness_value_now)
        self.contrastSlider.setSliderPosition(self.contrast_value_now)

    def histogramEqualization(self):
        listA, listB = calculate_histogram_list(self.imgA, self.imgB)
        cumsumA = np.zeros(256)
        cumsumB = np.zeros(256)
        cumsumA[0] = listA[0]
        cumsumB[0] = listB[0]
        for i in range(1, 256):
            cumsumA[i] = cumsumA[i-1] + listA[i]
            cumsumB[i] = cumsumB[i-1] + listB[i]

        mappingA = np.zeros(256, dtype=int)
        mappingB = np.zeros(256, dtype=int)
        L = 255
        for i in range(256):
            mappingA[i] = max(
                0, round((L*cumsumA[i])/(len(self.imgA)*len(self.imgA[0]))))
            mappingB[i] = max(
                0, round((L*cumsumB[i])/(len(self.imgA)*len(self.imgA[0]))))

        for i in range(len(self.imgA)):
            for j in range(len(self.imgA[0])):
                self.imgA[i][j] = mappingA[self.imgA[i][j]]
                self.imgB[i][j] = mappingB[self.imgB[i][j]]
        self.imgA = self.imgA.astype(np.uint8)
        self.imgB = self.imgB.astype(np.uint8)
        # self.showImage()
        self.showGrayImage(self.imgA, self.imgB)
        self.showBarChart(self.imgA, self.imgB)

    def modify_image(self):
        threshold = self.threshold.text()
        grayScaleLevels = self.grayscaleLevel.text()
        spatialResolution = self.spatialResolution.text()

        imgA = np.copy(self.imgA)
        imgB = np.copy(self.imgB)

        # # For Problem 4
        if threshold != '':
            threshold = int(threshold)
            for i in range(len(imgA)):
                for j in range(len(imgA[0])):
                    if self.imgA[i][j] >= threshold:
                        imgA[i][j] = 255
                    else:
                        imgA[i][j] = 0
                    if self.imgB[i][j] >= threshold:
                        imgB[i][j] = 255
                    else:
                        imgB[i][j] = 0

        # For Problem 5
        if grayScaleLevels != '':
            grayScaleLevels = int(grayScaleLevels)
            interval = 255 / grayScaleLevels
            scale = 255 / (grayScaleLevels - 1)

            for i in range(len(imgA)):
                for j in range(len(imgA[0])):
                    if imgA[i][j] == 0:
                        pass
                    elif imgA[i][j] % interval == 0:
                        s = imgA[i][j] / interval - 1
                        imgA[i][j] = s * scale
                    else:
                        s = int(imgA[i][j] / interval)
                        imgA[i][j] = s * scale

                    if imgB[i][j] == 0:
                        pass
                    elif imgB[i][j] % interval == 0:
                        s = imgB[i][j] / interval - 1
                        imgB[i][j] = s * scale
                    else:
                        s = int(imgB[i][j] / interval)
                        imgB[i][j] = s * scale

            imgA = imgA.astype(np.uint8)
            imgB = imgB.astype(np.uint8)

        # For Problem 5
        if spatialResolution != '':
            mu = (int(spatialResolution) + 10) / 10
            imgA = self.bilinear_interpolation(
                imgA, int(mu*len(imgA)), int(mu*len(imgA[0])))
            imgB = self.bilinear_interpolation(
                imgB, int(mu*len(imgB)), int(mu*len(imgB[0])))

        # For Problem 6
        if self.brightness_value_now != 0:
            if self.brightness_value_now > 0:
                shadow = self.brightness_value_now
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + self.brightness_value_now
            alpha = (highlight - shadow)/255
            gamma = shadow
            for i in range(len(imgA)):
                for j in range(len(imgA[0])):
                    imgA[i][j] = alpha * imgA[i][j] + gamma
                    imgB[i][j] = alpha * imgB[i][j] + gamma
            imgA = imgA.astype(np.uint8)
            imgB = imgB.astype(np.uint8)

        # For Problem 6
        if self.contrast_value_now != 0:
            f = 131*(self.contrast_value_now + 127) / \
                (127*(131-self.contrast_value_now))
            alpha = f
            gamma = 127*(1-f)
            for i in range(len(imgA)):
                for j in range(len(imgA[0])):
                    imgA[i][j] = alpha * imgA[i][j].astype(np.int32) + gamma
                    imgB[i][j] = alpha * imgB[i][j].astype(np.int32) + gamma
            imgA = imgA.astype(np.uint8)
            imgB = imgB.astype(np.uint8)

        # self.original.setPixmap(QtGui.QPixmap.fromImage(self.image))
        self.showGrayImage(imgA, imgB)
        self.showBarChart(imgA, imgB)
        # print(self.brightness_value_now,self.contrast_value_now)

    def bilinear_interpolation(self, image, x, y):
        height = image.shape[0]
        width = image.shape[1]
        scale_x = width/y
        scale_y = height/x
        new_image = np.zeros((x, y))
        for i in range(x):
            for j in range(y):
                x_int = int((j+0.5) * (scale_x) - 0.5)
                y_int = int((i+0.5) * (scale_y) - 0.5)

                x_int = min(x_int, width-2)
                y_int = min(y_int, height-2)

                x_diff = scale_x * j - x_int
                y_diff = scale_y * i - y_int
                a = image[y_int, x_int]
                b = image[y_int, x_int+1]
                c = image[y_int+1, x_int]
                d = image[y_int+1, x_int+1]

                pixel = a*(1-x_diff)*(1-y_diff) + b*(x_diff) * \
                    (1-y_diff) + c*(1-x_diff) * (y_diff) + d*x_diff*y_diff
                new_image[i, j] = pixel.astype(np.uint8)
        return new_image

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.loadButton.setText(_translate("MainWindow", "Load"))
        self.autoAdjustment.setText(
            _translate("MainWindow", "Auto adjustment"))
        self.confirmButton.setText(_translate("MainWindow", "confirm"))
        self.thresholdLabel.setText(_translate(
            "MainWindow", "threshold to binary image(0~255)"))
        self.label_4.setText(_translate(
            "MainWindow", "spatial resolution (-9~9)"))
        self.label_5.setText(_translate(
            "MainWindow", "Grayscale level (2~256)"))
        self.label_2.setText(_translate("MainWindow", "brightness (-99~99)"))
        self.label_3.setText(_translate("MainWindow", "contrast (-99~99)"))
        self.BrightnessSlider.setSliderPosition(self.brightness_value_now)
        self.contrastSlider.setSliderPosition(self.contrast_value_now)


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
