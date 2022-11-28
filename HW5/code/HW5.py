# %%
import sys
import cv2
import math
import matplotlib as plt
import numpy as np
from PyQt5 import QtGui, QtWidgets
from HW5layout import Ui_Form
from PyQt5.QtCore import Qt


class HW5(QtWidgets.QDialog):
    grayTable = cv2.imread('grayTable.png', 1)
    grayTable = cv2.cvtColor(grayTable,cv2.COLOR_BGR2RGB)
    nameList = ["autumn","bone","jet","winter","rainbow","ocean","summer","spring","cool","HSV","pink","hot","parula","magma","inferno","plasma","viridis","cividis","twilight","twilight shifted","turbo","deepgreen"]
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.load.clicked.connect(self.loadImage)
        self.ui.clean.clicked.connect(self.refreshImage)
        self.ui.p1Confirm.clicked.connect(self.colorConversion)
        # self.ui.p2Confirm.clicked.connect(self.colorMapping)
        self.ui.p3Confirm.clicked.connect(self.colorSegmentation)
        self.ui.mapNumber.valueChanged['int'].connect(self.changeMap)
        self.ui.mapNumber.valueChanged.connect(lambda:self.boxChange())
        self.ui.showNumber.valueChanged.connect(lambda:self.sliderChange())
        self.show()
    def boxChange(self):
      self.ui.showNumber.setValue(self.ui.mapNumber.value())
    def sliderChange(self):
      self.ui.mapNumber.setValue(self.ui.showNumber.value())
    def changeMap(self,value):
      self.ui.mapType.setText(self.nameList[value])
      self.refreshImage()
      img_gray = cv2.cvtColor(self.image.copy(), cv2.COLOR_RGB2GRAY)
      img_color = cv2.applyColorMap(img_gray, value)
      # grayTable = cv2.imread('grayTable.png', 1)
      colorTable = cv2.applyColorMap(self.grayTable, value)
      self.showImage(self.image,self.ui.originalImage)
      self.showImage(self.grayTable,self.ui.modifiedImage2)
      self.showImage(colorTable,self.ui.modifiedImage3)
      self.showImage(img_color, self.ui.modifiedImage1)

    def loadImage(self):
      self.refreshImage()
      filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Open Image', 'Image', '*.png *.jpg *.bmp')
      if filename == '':
          return
      # Load color image
      self.image = cv2.imread(filename, -1)
      # BGR to RGB
      self.image = cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)
      self.showImage(self.image,self.ui.originalImage)
      
    def showImage(self,imageToShow,block):
      if len(imageToShow.shape) == 3:
        height,width ,color = imageToShow.shape
        outputImage = QtGui.QImage(imageToShow.astype(
            np.uint8), width, height, 3 * width, QtGui.QImage.Format_RGB888)
      else:
        height,width = imageToShow.shape
        outputImage = QtGui.QImage(imageToShow.astype(
            np.uint8), width, height,width, QtGui.QImage.Format_Grayscale8)
      pixmap = QtGui.QPixmap(outputImage)
      # width = 380 , height= 285
      pixmap = pixmap.scaled(380, 380, aspectRatioMode=Qt.KeepAspectRatio)
      block.setPixmap(pixmap)
    def colorConversion(self):
      self.refreshImage()
      imageCopy = self.image.copy().astype(np.float64)
      imageCMY = self.RGB2CMY(imageCopy/255)
      imageHSI = self.RGB2HSI(imageCopy)
      imageXYZ = self.RGB2XYZ(imageCopy/255)
      imageLab = self.XYZ2Lab(imageXYZ)
      imageYUV = self.RGB2YUV(imageCopy/255)
      self.showImage(self.image,self.ui.originalImage)
      self.showImage(imageCMY*255,self.ui.modifiedImage1)
      self.showImage(imageHSI,self.ui.modifiedImage2)
      self.showImage(self.convertTo0_255(imageXYZ*255),self.ui.modifiedImage3)
      self.showImage(self.convertTo0_255(imageLab*255),self.ui.modifiedImage4)
      self.showImage(self.convertTo0_255(imageYUV*255),self.ui.modifiedImage5)
    def RGB2YUV(self,inputImage):
      rows,cols,color = inputImage.shape
      YUV_w = np.array([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]])
      YUV_b = np.array([0, 128, 128]) / 256

      imageYUV = np.zeros((rows, cols, 3))
      for i in range(rows):
          for j in range(cols):
              imageYUV[i][j] = np.dot(YUV_w, inputImage[i][j]) + YUV_b

      return imageYUV

    def XYZ2Lab(self,inputImage):
      def h(q):
        if q > 0.008856: 
          return q**(1/3)
        else:
          return 7.787*q +16/116
      rows,cols,color = inputImage.shape
      Xn = 0.95047
      Yn = 1.0
      Zn = 1.08883
      L= np.zeros((rows,cols))
      a= np.zeros((rows,cols))
      b= np.zeros((rows,cols))
      for i in range(rows):
        for j in range(cols):
          L[i,j] = 116 * h(inputImage[i,j,1]/Yn) - 16
          a[i,j] = 500 * (h(inputImage[i,j,0]/Xn) - h(inputImage[i,j,1]/Yn))
          b[i,j] = 200 * (h(inputImage[i,j,1]/Yn) - h(inputImage[i,j,2]/Zn))

      imageLab = np.zeros((rows, cols, 3))
      imageLab[:,:,0] = L
      imageLab[:,:,1] = a
      imageLab[:,:,2] = b
      return imageLab

    def RGB2XYZ(self,inputImage):
      rows,cols,color = inputImage.shape
      XYZ_w = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
      imageXYZ = np.zeros((rows, cols, 3))
      for i in range(rows):
          for j in range(cols):
              imageXYZ[i][j] = np.dot(XYZ_w, inputImage[i,j])

      return imageXYZ

    def RGB2HSI(self,inputImage):
      rows,cols,color = inputImage.shape
      B, G, R = cv2.split(inputImage)
      # 歸一化到[0,1]
      B = B / 255.0
      G = G / 255.0
      R = R / 255.0
      imageHSI = inputImage.copy()
      H, S, I = cv2.split(imageHSI)
      for i in range(rows):
          for j in range(cols):
              num = 0.5 * ((R[i, j] - G[i, j]) + (R[i, j] - B[i, j]))
              den = np.sqrt((R[i, j] - G[i, j]) ** 2 + (R[i, j] - B[i, j]) * (G[i, j] - B[i, j]))
              theta = float(np.arccos(num / den))

              if den == 0:
                  H = 0
              elif B[i, j] <= G[i, j]:
                  H = theta
              else:
                  H = 2 * math.pi - theta

              min_RGB = min(min(B[i, j], G[i, j]), R[i, j])
              sum = B[i, j] + G[i, j] + R[i, j]
              if sum == 0:
                  S = 0
              else:
                  S = 1 - 3 * min_RGB / sum

              H = H / (2 * math.pi)
              I = sum / 3.0
              # 輸出HSI圖像，擴充到255以方便顯示，一般H分量在[0,2pi]之間，S和I在[0,1]之間
              imageHSI[i, j, 0] = H * 255
              imageHSI[i, j, 1] = S * 255
              imageHSI[i, j, 2] = I * 255
      return imageHSI
    
    def RGB2CMY(self,inputImage):
      rows,cols,color = inputImage.shape
      C = 1 - inputImage[:,:,0]
      M = 1 - inputImage[:,:,1]
      Y = 1 - inputImage[:,:,2]

      imageCMY = np.zeros((rows, cols, 3))
      imageCMY[:,:,0] = C
      imageCMY[:,:,1] = M
      imageCMY[:,:,2] = Y
      return imageCMY
    def colorSegmentation(self):
      self.refreshImage()
      k  = int(self.ui.p3K.text())
      maxIter = int(self.ui.p3Max.text())
      epsilon = float(self.ui.p3Epsilon.text())
      imageCopy = self.image.copy().astype(np.float32)
      imageHSI = self.RGB2HSI(imageCopy).astype(np.float32)
      imageLab = self.XYZ2Lab(self.RGB2XYZ(imageCopy/255))
      imageLab = (imageLab*255).astype(np.float32)
      imageOutput1 = self.kMean(imageCopy,k,maxIter,epsilon)
      imageOutput2 = self.kMean(imageHSI,k,maxIter,epsilon)
      imageOutput3 = self.kMean(imageLab,k,maxIter,epsilon)
      self.showImage(self.image,self.ui.originalImage)
      self.showImage(imageHSI,self.ui.modifiedImage2)
      self.showImage(self.convertTo0_255(imageLab),self.ui.modifiedImage4)
      self.showImage(255-self.convertTo0_255(imageOutput1),self.ui.modifiedImage1)
      self.showImage(255-self.convertTo0_255(imageOutput2),self.ui.modifiedImage3)
      self.showImage(self.convertTo0_255(imageOutput3),self.ui.modifiedImage5)

    def colorMapping(self):
      self.refreshImage()
      img_gray = cv2.cvtColor(self.image.copy(), cv2.COLOR_RGB2GRAY)
      custom_cmap = plt.colors.LinearSegmentedColormap.from_list("custom",["red" , "yellow"])
      img_color = cv2.applyColorMap(img_gray, 21)
      self.showImage(img_color, self.ui.modifiedImage1)


    def kMean(self,inputImage,k,maxIter, epsilon):
      rows,cols,color = inputImage.shape
      inputImage = inputImage.reshape((rows*cols, color))
      # Define criteria
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, maxIter, epsilon)
      # Set flags
      flags = cv2.KMEANS_RANDOM_CENTERS
      # Apply KMeans
      compactness,labels,centers = cv2.kmeans(inputImage, k, None, criteria, 10, flags)
      returnImage = labels.reshape((rows,cols))
      return returnImage


    def refreshImage(self):
      imageCopy = self.grayTable.copy().astype(np.float64)
      self.showImage(imageCopy-imageCopy+236, self.ui.originalImage)
      self.showImage(imageCopy-imageCopy+236, self.ui.modifiedImage1)
      self.showImage(imageCopy-imageCopy+236, self.ui.modifiedImage2)
      self.showImage(imageCopy-imageCopy+236, self.ui.modifiedImage3)
      self.showImage(imageCopy-imageCopy+236, self.ui.modifiedImage4)
      self.showImage(imageCopy-imageCopy+236, self.ui.modifiedImage5)
    def convertTo0_255(self, inputImage):
        returnImage = inputImage.copy()
        returnImage = np.round(
            (returnImage - returnImage.min()) / (returnImage.max() - returnImage.min()) * 255)
        return returnImage.astype(np.uint8)
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = HW5()
    ui.show()
    sys.exit(app.exec_())


