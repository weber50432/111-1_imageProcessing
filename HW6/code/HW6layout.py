# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'HW6_layout.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1184, 941)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setObjectName("verticalLayout")
        self.controlLayout = QtWidgets.QGridLayout()
        self.controlLayout.setHorizontalSpacing(6)
        self.controlLayout.setObjectName("controlLayout")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setMinimumSize(QtCore.QSize(50, 0))
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.controlLayout.addWidget(self.label_3, 1, 0, 1, 2)
        self.label_7 = QtWidgets.QLabel(Form)
        self.label_7.setMinimumSize(QtCore.QSize(150, 0))
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.controlLayout.addWidget(self.label_7, 1, 6, 1, 1)
        self.p3Confirm = QtWidgets.QPushButton(Form)
        self.p3Confirm.setMinimumSize(QtCore.QSize(75, 0))
        self.p3Confirm.setObjectName("p3Confirm")
        self.controlLayout.addWidget(self.p3Confirm, 4, 3, 1, 1)
        self.p2Confirm = QtWidgets.QPushButton(Form)
        self.p2Confirm.setMinimumSize(QtCore.QSize(75, 0))
        self.p2Confirm.setObjectName("p2Confirm")
        self.controlLayout.addWidget(self.p2Confirm, 2, 3, 1, 1)
        self.p1_x = QtWidgets.QLineEdit(Form)
        self.p1_x.setMinimumSize(QtCore.QSize(0, 0))
        self.p1_x.setObjectName("p1_x")
        self.controlLayout.addWidget(self.p1_x, 1, 5, 1, 1)
        self.label_9 = QtWidgets.QLabel(Form)
        self.label_9.setMinimumSize(QtCore.QSize(50, 0))
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.controlLayout.addWidget(self.label_9, 4, 0, 1, 2)
        self.p1Confirm = QtWidgets.QPushButton(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.p1Confirm.sizePolicy().hasHeightForWidth())
        self.p1Confirm.setSizePolicy(sizePolicy)
        self.p1Confirm.setMinimumSize(QtCore.QSize(100, 0))
        self.p1Confirm.setObjectName("p1Confirm")
        self.controlLayout.addWidget(self.p1Confirm, 1, 3, 1, 1)
        self.p2Load = QtWidgets.QPushButton(Form)
        self.p2Load.setMinimumSize(QtCore.QSize(100, 0))
        self.p2Load.setObjectName("p2Load")
        self.controlLayout.addWidget(self.p2Load, 2, 4, 1, 1)
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setMinimumSize(QtCore.QSize(50, 0))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.controlLayout.addWidget(self.label_2, 2, 0, 1, 2)
        self.label_8 = QtWidgets.QLabel(Form)
        self.label_8.setMinimumSize(QtCore.QSize(150, 0))
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.controlLayout.addWidget(self.label_8, 4, 4, 1, 1)
        self.load = QtWidgets.QPushButton(Form)
        self.load.setMinimumSize(QtCore.QSize(100, 0))
        self.load.setObjectName("load")
        self.controlLayout.addWidget(self.load, 0, 0, 1, 2)
        self.p1_y = QtWidgets.QLineEdit(Form)
        self.p1_y.setMinimumSize(QtCore.QSize(0, 0))
        self.p1_y.setObjectName("p1_y")
        self.controlLayout.addWidget(self.p1_y, 1, 7, 1, 1)
        self.label = QtWidgets.QLabel(Form)
        self.label.setMinimumSize(QtCore.QSize(150, 0))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.controlLayout.addWidget(self.label, 1, 4, 1, 1)
        self.mapType = QtWidgets.QLabel(Form)
        self.mapType.setText("")
        self.mapType.setAlignment(QtCore.Qt.AlignCenter)
        self.mapType.setObjectName("mapType")
        self.controlLayout.addWidget(self.mapType, 0, 7, 1, 2)
        self.clean = QtWidgets.QPushButton(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clean.sizePolicy().hasHeightForWidth())
        self.clean.setSizePolicy(sizePolicy)
        self.clean.setMinimumSize(QtCore.QSize(100, 0))
        self.clean.setObjectName("clean")
        self.controlLayout.addWidget(self.clean, 0, 3, 1, 1)
        self.label_10 = QtWidgets.QLabel(Form)
        self.label_10.setMinimumSize(QtCore.QSize(150, 0))
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.controlLayout.addWidget(self.label_10, 4, 6, 1, 1)
        self.p3Areas = QtWidgets.QLabel(Form)
        self.p3Areas.setMinimumSize(QtCore.QSize(150, 0))
        self.p3Areas.setText("")
        self.p3Areas.setAlignment(QtCore.Qt.AlignCenter)
        self.p3Areas.setObjectName("p3Areas")
        self.controlLayout.addWidget(self.p3Areas, 4, 5, 1, 1)
        self.p3Perimeters = QtWidgets.QLabel(Form)
        self.p3Perimeters.setMinimumSize(QtCore.QSize(150, 0))
        self.p3Perimeters.setText("")
        self.p3Perimeters.setAlignment(QtCore.Qt.AlignCenter)
        self.p3Perimeters.setObjectName("p3Perimeters")
        self.controlLayout.addWidget(self.p3Perimeters, 4, 7, 1, 1)
        self.verticalLayout.addLayout(self.controlLayout)
        self.imageLayout = QtWidgets.QGridLayout()
        self.imageLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.imageLayout.setObjectName("imageLayout")
        self.modifiedImage3 = QtWidgets.QLabel(Form)
        self.modifiedImage3.setMinimumSize(QtCore.QSize(380, 380))
        self.modifiedImage3.setText("")
        self.modifiedImage3.setObjectName("modifiedImage3")
        self.imageLayout.addWidget(self.modifiedImage3, 1, 1, 1, 1)
        self.modifiedImage2 = QtWidgets.QLabel(Form)
        self.modifiedImage2.setMinimumSize(QtCore.QSize(380, 380))
        self.modifiedImage2.setText("")
        self.modifiedImage2.setObjectName("modifiedImage2")
        self.imageLayout.addWidget(self.modifiedImage2, 1, 0, 1, 1)
        self.modifiedImage1_3 = QtWidgets.QLabel(Form)
        self.modifiedImage1_3.setMinimumSize(QtCore.QSize(380, 380))
        self.modifiedImage1_3.setText("")
        self.modifiedImage1_3.setObjectName("modifiedImage1_3")
        self.imageLayout.addWidget(self.modifiedImage1_3, 0, 2, 1, 1)
        self.originalImage = QtWidgets.QLabel(Form)
        self.originalImage.setMinimumSize(QtCore.QSize(380, 380))
        self.originalImage.setText("")
        self.originalImage.setObjectName("originalImage")
        self.imageLayout.addWidget(self.originalImage, 0, 0, 1, 1)
        self.modifiedImage1 = QtWidgets.QLabel(Form)
        self.modifiedImage1.setMinimumSize(QtCore.QSize(380, 380))
        self.modifiedImage1.setText("")
        self.modifiedImage1.setObjectName("modifiedImage1")
        self.imageLayout.addWidget(self.modifiedImage1, 0, 1, 1, 1)
        self.verticalLayout.addLayout(self.imageLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_3.setText(_translate("Form", "Part 1:"))
        self.label_7.setText(_translate("Form", "y_offset:"))
        self.p3Confirm.setText(_translate("Form", "P3Confirm"))
        self.p2Confirm.setText(_translate("Form", "P2Confirm"))
        self.label_9.setText(_translate("Form", "Part 3:"))
        self.p1Confirm.setText(_translate("Form", "P1Confirm"))
        self.p2Load.setText(_translate("Form", "LoadMany"))
        self.label_2.setText(_translate("Form", "Part 2:"))
        self.label_8.setText(_translate("Form", "areas:"))
        self.load.setText(_translate("Form", "Load"))
        self.label.setText(_translate("Form", "x_offset:"))
        self.clean.setText(_translate("Form", "Clean"))
        self.label_10.setText(_translate("Form", "perimeters:"))
